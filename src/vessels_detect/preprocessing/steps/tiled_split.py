"""
src/vessels_detect/preprocessing/steps/tiled_split.py
------------------------------------------------------
Stage 6 - Tile-first Split with Sub-folder Sharding.

New pipeline strategy
~~~~~~~~~~~~~~~~~~~~~
Instead of splitting raw images before tiling (old Stages 4→5), this step:

1. **Tiles** every processed GeoTIFF in ``spatial_dir`` (or ``raw_dir`` if
   spatial is skipped) into non-overlapping fixed-size patches with their
   matching YOLO OBB label files, storing all tiles in a flat staging area.
2. **Splits** the full tile pool into ``train`` (80 %) and ``val`` (20 %)
   using a per-source-image interleaving strategy so every original image
   contributes tiles to both subsets.
3. **Shards** the train set into 8 numbered sub-folders
   (``train/1`` … ``train/8``), again interleaving tiles from every source
   image across all 8 shards so each shard is a representative mix.
4. **Writes** a ``metadata.csv`` alongside the output root summarising, for
   every folder, the percentage of tiles from each source image, the total
   object count per class, and the total tile count.

Output layout
~~~~~~~~~~~~~
::

    <output_dir>/           (cfg["paths"]["tiled_split_dir"])
      train/
        1/
          images/  *.tif
          labels/  *.txt
        2/ … 8/  (same layout)
      val/
        images/  *.tif
        labels/  *.txt
      metadata.csv

Configuration (``cfg["tiled_split"]``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    tiled_split:
      tile_size:        2048
      compress:         lzw
      min_visible_frac: 0.10
      val_ratio:        0.20
      n_train_shards:   8
      random_seed:      42
      copy:             true   # true → copy tiles, false → hard-link / copy
"""

from __future__ import annotations

import csv
import logging
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from shapely.geometry import Polygon

from src.vessels_detect.preprocessing.steps.base import BaseStep
from src.vessels_detect.preprocessing.steps.tiling import (
    tile_image_raw,
    project_labels_to_tile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TiledSplitConfig:
    """All hyperparameters consumed by :class:`TiledSplitStep`.

    Attributes:
        tile_size: Output tile width and height in pixels.
        compress: GeoTIFF compression codec.
        min_visible_frac: Minimum fraction of an OBB visible inside a tile
            for the label line to be kept.
        val_ratio: Fraction of tiles from each source image assigned to
            validation.  Remaining tiles go to train.
        n_train_shards: Number of numbered sub-folders inside ``train/``.
        random_seed: Seed for reproducible shuffling.
        copy: If ``True``, tiles are copied to the destination directories.
            If ``False``, tiles are moved (saves disk space).
    """

    tile_size:        int   = 2048
    compress:         str   = "lzw"
    min_visible_frac: float = 0.10
    val_ratio:        float = 0.20
    n_train_shards:   int   = 8
    random_seed:      int   = 42
    copy:             bool  = True

    def __post_init__(self) -> None:
        if not 0.0 < self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}.")
        if self.n_train_shards < 1:
            raise ValueError(f"n_train_shards must be ≥ 1, got {self.n_train_shards}.")

    @classmethod
    def from_dict(cls, cfg: dict) -> "TiledSplitConfig":
        return cls(
            tile_size=int(cfg.get("tile_size", 2048)),
            compress=str(cfg.get("compress", "lzw")),
            min_visible_frac=float(cfg.get("min_visible_frac", 0.10)),
            val_ratio=float(cfg.get("val_ratio", 0.20)),
            n_train_shards=int(cfg.get("n_train_shards", 8)),
            random_seed=int(cfg.get("random_seed", 42)),
            copy=bool(cfg.get("copy", True)),
        )


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _read_label_profile(label_path: Path) -> Dict[int, int]:
    """Count per-class instances in one YOLO label file."""
    profile: Dict[int, int] = defaultdict(int)
    if not label_path.exists():
        return dict(profile)
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                cls_id = int(line.split()[0])
                profile[cls_id] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot parse label '%s': %s", label_path.name, exc)
    return dict(profile)


# ---------------------------------------------------------------------------
# Transfer helper
# ---------------------------------------------------------------------------

def _transfer(src: Path, dst: Path, copy: bool) -> None:
    """Copy or move *src* to *dst*, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.debug("Source does not exist, skipping: %s", src)
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


# ---------------------------------------------------------------------------
# Metadata CSV writer
# ---------------------------------------------------------------------------

def _write_metadata(
    output_dir: Path,
    folder_records: Dict[str, List[Tuple[Path, Path]]],
    all_source_stems: List[str],
) -> None:
    """Write ``metadata.csv`` into *output_dir*.

    For every output folder the CSV records:
    - ``folder``: relative path inside *output_dir* (e.g. ``train/1``, ``val``).
    - ``n_tiles``: total tile count.
    - ``pct_<source_stem>``: percentage of tiles originating from that source.
    - ``n_class_<id>``: total object count for each YOLO class present.

    Args:
        output_dir: Root output directory (where the CSV is written).
        folder_records: Mapping from folder label to list of
            ``(image_tile_path, label_tile_path)`` tuples.
        all_source_stems: Sorted list of all source image stem names.
    """
    # Collect all class IDs across every label file.
    all_class_ids: set = set()
    for records in folder_records.values():
        for _img, lbl in records:
            for cls_id in _read_label_profile(lbl):
                all_class_ids.add(cls_id)
    sorted_classes = sorted(all_class_ids)

    rows = []
    for folder_name, records in sorted(folder_records.items()):
        n_tiles = len(records)

        # Count tiles per source stem.
        source_counts: Dict[str, int] = defaultdict(int)
        class_counts:  Dict[int, int] = defaultdict(int)

        for img_path, lbl_path in records:
            # Tile name format: {source_stem}_{x_off}_{y_off}.tif
            # The stem may itself contain underscores, so we extract the
            # source by stripping the last two "_<int>" tokens.
            parts = img_path.stem.rsplit("_", 2)
            src_stem = parts[0] if len(parts) == 3 else img_path.stem
            source_counts[src_stem] += 1

            for cls_id, cnt in _read_label_profile(lbl_path).items():
                class_counts[cls_id] += cnt

        row: dict = {"folder": folder_name, "n_tiles": n_tiles}
        for stem in all_source_stems:
            pct = (source_counts.get(stem, 0) / n_tiles * 100) if n_tiles else 0.0
            row[f"pct_{stem}"] = f"{pct:.2f}"
        for cls_id in sorted_classes:
            row[f"n_class_{cls_id}"] = class_counts.get(cls_id, 0)
        rows.append(row)

    if not rows:
        logger.warning("No rows to write in metadata.csv.")
        return

    fieldnames = ["folder", "n_tiles"]
    for stem in all_source_stems:
        fieldnames.append(f"pct_{stem}")
    for cls_id in sorted_classes:
        fieldnames.append(f"n_class_{cls_id}")

    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Metadata written → %s", csv_path)


# ---------------------------------------------------------------------------
# Core split / shard logic
# ---------------------------------------------------------------------------

def _interleave_split(
    tile_records: List[Tuple[Path, Path]],
    val_ratio: float,
    rng: random.Random,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Split tile records from ONE source image into train / val subsets.

    Tiles are shuffled then allocated by interleaving so that every
    ``round(1 / val_ratio)``-th tile goes to val and the rest to train.
    This gives each source image a proportional representation in both sets.

    Args:
        tile_records: List of ``(image_tile_path, label_tile_path)`` tuples.
        val_ratio: Target fraction to assign to validation.
        rng: Seeded random instance.

    Returns:
        ``(train_records, val_records)`` tuple.
    """
    shuffled = tile_records[:]
    rng.shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_ratio)) if shuffled else 0
    val_records   = shuffled[:n_val]
    train_records = shuffled[n_val:]
    return train_records, val_records


def _shard_tiles(
    train_records: List[Tuple[Path, Path]],
    n_shards: int,
    rng: random.Random,
) -> Dict[int, List[Tuple[Path, Path]]]:
    """Distribute train tiles across *n_shards* numbered buckets.

    The distribution is done per source image (tiles from each source are
    spread evenly across all shards before moving to the next source) so
    every shard receives a balanced mix of source images.

    Args:
        train_records: All train tile records (from all source images).
        n_shards: Number of shards to create.
        rng: Seeded random instance for within-source shuffling.

    Returns:
        Mapping ``{shard_index (1-based): list_of_records}``.
    """
    # Group by source stem.
    by_source: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    for rec in train_records:
        img_path = rec[0]
        parts = img_path.stem.rsplit("_", 2)
        src_stem = parts[0] if len(parts) == 3 else img_path.stem
        by_source[src_stem].append(rec)

    shards: Dict[int, List[Tuple[Path, Path]]] = {i + 1: [] for i in range(n_shards)}

    for src_stem, recs in by_source.items():
        shuffled = recs[:]
        rng.shuffle(shuffled)
        for i, rec in enumerate(shuffled):
            shard_idx = (i % n_shards) + 1  # 1-based
            shards[shard_idx].append(rec)

    return shards


# ---------------------------------------------------------------------------
# Main step
# ---------------------------------------------------------------------------

class TiledSplitStep(BaseStep):
    """Stage 6 - tile-first split with train sub-folder sharding.

    Tiles all processed GeoTIFFs from ``spatial_dir`` into non-overlapping
    patches, then splits and shards the tile pool into::

        <tiled_split_dir>/
          train/{1…8}/images/*.tif
          train/{1…8}/labels/*.txt
          val/images/*.tif
          val/labels/*.txt
          metadata.csv
    """

    NAME = "tiled_split"

    def run(self, cfg: dict) -> None:  # noqa: C901
        """Execute the tiled-split stage.

        Args:
            cfg: Fully resolved configuration dictionary.
        """
        paths  = cfg["paths"]
        params = TiledSplitConfig.from_dict(cfg.get("tiled_split", {}))

        # Source: prefer spatial_dir if it has .tif files, otherwise raw_dir.
        for src_key in ("spatial_dir", "radiometric_dir", "raw_dir"):
            src_dir: Path = paths[src_key]
            tif_files = sorted(src_dir.glob("*.tif")) if src_dir.exists() else []
            if tif_files:
                logger.info("Source directory: %s (%s)", src_dir, src_key)
                break
        else:
            raise RuntimeError(
                "No .tif files found in spatial_dir, radiometric_dir, or raw_dir."
            )

        labels_dir: Path  = paths["labels_dir"]
        output_dir: Path  = paths["tiled_split_dir"]
        staging_dir: Path = output_dir / "_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        logger.info("TiledSplit configuration:")
        logger.info("  tile_size        : %d px", params.tile_size)
        logger.info("  overlap          : 0 (non-overlapping)")
        logger.info("  val_ratio        : %.2f", params.val_ratio)
        logger.info("  n_train_shards   : %d", params.n_train_shards)
        logger.info("  min_visible_frac : %.2f", params.min_visible_frac)
        logger.info("  compress         : %s", params.compress)
        logger.info("  output_dir       : %s", output_dir)

        rng = random.Random(params.random_seed)

        # ── Stage 1: Tile every source image into the staging directory ──────
        all_tile_records: Dict[str, List[Tuple[Path, Path]]] = {}  # stem → tiles

        for tif_path in tif_files:
            stem = tif_path.stem
            stg_img = staging_dir / "images"
            stg_lbl = staging_dir / "labels"
            stg_img.mkdir(parents=True, exist_ok=True)
            stg_lbl.mkdir(parents=True, exist_ok=True)

            logger.info("  Tiling: %s", tif_path.name)

            with rasterio.open(tif_path) as src:
                img_w, img_h = src.width, src.height

            tile_records_raw = tile_image_raw(
                tif_path=tif_path,
                output_image_dir=stg_img,
                tile_size=params.tile_size,
                overlap=0,               # ← non-overlapping
                compress=params.compress,
            )

            label_path = labels_dir / f"{stem}.txt"
            if not label_path.exists():
                label_path = None

            tile_pairs: List[Tuple[Path, Path]] = []
            for tile_img_path, x_off, y_off in tile_records_raw:
                label_lines = project_labels_to_tile(
                    label_path=label_path,
                    img_width=img_w,
                    img_height=img_h,
                    x_off=x_off,
                    y_off=y_off,
                    tile_size=params.tile_size,
                    min_visible_frac=params.min_visible_frac,
                )
                lbl_path = stg_lbl / f"{tile_img_path.stem}.txt"
                lbl_path.write_text("\n".join(label_lines), encoding="utf-8")
                tile_pairs.append((tile_img_path, lbl_path))

            all_tile_records[stem] = tile_pairs
            logger.info(
                "    → %d tile(s) from %s", len(tile_pairs), tif_path.name
            )

        # ── Stage 2: Per-source train / val split ─────────────────────────
        all_train: List[Tuple[Path, Path]] = []
        all_val:   List[Tuple[Path, Path]] = []

        for stem, tile_pairs in all_tile_records.items():
            train_recs, val_recs = _interleave_split(tile_pairs, params.val_ratio, rng)
            all_train.extend(train_recs)
            all_val.extend(val_recs)
            logger.info(
                "  %-40s → train=%d  val=%d",
                stem, len(train_recs), len(val_recs),
            )

        logger.info(
            "Split totals: train=%d  val=%d  (total=%d)",
            len(all_train), len(all_val), len(all_train) + len(all_val),
        )

        # ── Stage 3: Shard train tiles across n_train_shards buckets ─────
        shards = _shard_tiles(all_train, params.n_train_shards, rng)
        for shard_idx, recs in shards.items():
            logger.info("  train/%d : %d tile(s)", shard_idx, len(recs))

        # ── Stage 4: Write files to final destinations ─────────────────────
        # Collect folder records for metadata.
        folder_records: Dict[str, List[Tuple[Path, Path]]] = {}

        # val
        val_dir_img = output_dir / "val" / "images"
        val_dir_lbl = output_dir / "val" / "labels"
        val_dir_img.mkdir(parents=True, exist_ok=True)
        val_dir_lbl.mkdir(parents=True, exist_ok=True)

        val_final: List[Tuple[Path, Path]] = []
        for src_img, src_lbl in all_val:
            dst_img = val_dir_img / src_img.name
            dst_lbl = val_dir_lbl / src_lbl.name
            _transfer(src_img, dst_img, params.copy)
            _transfer(src_lbl, dst_lbl, params.copy)
            val_final.append((dst_img, dst_lbl))
        folder_records["val"] = val_final

        # train shards
        for shard_idx, recs in shards.items():
            shard_img_dir = output_dir / "train" / str(shard_idx) / "images"
            shard_lbl_dir = output_dir / "train" / str(shard_idx) / "labels"
            shard_img_dir.mkdir(parents=True, exist_ok=True)
            shard_lbl_dir.mkdir(parents=True, exist_ok=True)

            shard_final: List[Tuple[Path, Path]] = []
            for src_img, src_lbl in recs:
                dst_img = shard_img_dir / src_img.name
                dst_lbl = shard_lbl_dir / src_lbl.name
                _transfer(src_img, dst_img, params.copy)
                _transfer(src_lbl, dst_lbl, params.copy)
                shard_final.append((dst_img, dst_lbl))
            folder_records[f"train/{shard_idx}"] = shard_final

        # ── Stage 5: Write metadata CSV ────────────────────────────────────
        all_source_stems = sorted(all_tile_records.keys())
        _write_metadata(output_dir, folder_records, all_source_stems)

        # ── Cleanup staging directory ──────────────────────────────────────
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
            logger.debug("Removed staging directory: %s", staging_dir)

        logger.info("TiledSplit stage complete → '%s'.", output_dir)