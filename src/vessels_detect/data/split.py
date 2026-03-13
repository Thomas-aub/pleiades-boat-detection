"""
src/vessels_detect/data/split.py
-----------------
Splits tiles into train / val / test partitions with zero spatial leakage,
then optionally balances background (empty) tiles within each partition.

**Spatial Leakage Prevention**
    Tiles derived from the same source GeoTIFF spatially overlap and share
    geographic context.  Mixing them across partitions would allow the model
    to see validation or test regions during training.  This module solves
    the problem by grouping tiles at the *source-image level*: every tile
    from a given source ends up in exactly one partition.

    The source image is identified from the ``source_tif`` TIFF tag written
    by :mod:`src.data.tiler` — no filename parsing is required.

**Background Balancing**
    Satellite imagery of open water produces a large number of empty tiles.
    :class:`BackgroundBalancer` moves excess background tiles to an archive
    directory instead of deleting them, so they can be recovered if needed.
    The target background fraction is controlled by ``background_ratio``.

Typical usage::

    from src.data.split import DatasetSplitter, SplitConfig
    from src.data.split import BackgroundBalancer, BalanceConfig

    splitter = DatasetSplitter(SplitConfig(train_ratio=0.7, val_ratio=0.2))
    splitter.split(
        tiles_dir=Path("data/processed/tiles"),
        labels_raw_dir=Path("data/processed/labels_raw"),
        images_dir=Path("data/processed/images"),
        labels_dir=Path("data/processed/labels"),
    )

    balancer = BackgroundBalancer(BalanceConfig(background_ratio=0.10))
    balancer.balance(
        images_dir=Path("data/processed/images"),
        labels_dir=Path("data/processed/labels"),
        archive_dir=Path("data/processed/archive"),
        splits=["train", "val"],
    )
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import rasterio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Hyperparameters for :class:`DatasetSplitter`.

    Attributes:
        train_ratio: Fraction of source images assigned to the training set.
        val_ratio: Fraction assigned to the validation set.
        test_ratio: Fraction assigned to the test set.
            The three ratios must sum to 1.0 within floating-point tolerance.
        random_seed: Seed for reproducible shuffling of source images.
    """

    train_ratio: float = 0.70
    val_ratio: float   = 0.20
    test_ratio: float  = 0.10
    random_seed: int   = 42

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, got {total:.6f}."
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "SplitConfig":
        """Construct from a plain dictionary (e.g., parsed YAML section).

        Args:
            cfg: Dictionary with keys matching the dataclass field names.

        Returns:
            A populated :class:`SplitConfig` instance.
        """
        return cls(
            train_ratio=cfg.get("train_ratio", 0.70),
            val_ratio=cfg.get("val_ratio", 0.20),
            test_ratio=cfg.get("test_ratio", 0.10),
            random_seed=cfg.get("random_seed", 42),
        )


@dataclass
class BalanceConfig:
    """Hyperparameters for :class:`BackgroundBalancer`.

    Attributes:
        background_ratio: Target fraction of total tiles in a split that
            should be empty (background) tiles.  For example, ``0.10``
            keeps at most 10 % background after balancing.
        random_seed: Seed for reproducible selection of which background
            tiles to archive.
    """

    background_ratio: float = 0.10
    random_seed: int        = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.background_ratio < 1.0:
            raise ValueError(
                f"background_ratio must be in (0, 1), got {self.background_ratio}."
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "BalanceConfig":
        """Construct from a plain dictionary (e.g., parsed YAML section).

        Args:
            cfg: Dictionary with keys matching the dataclass field names.

        Returns:
            A populated :class:`BalanceConfig` instance.
        """
        return cls(
            background_ratio=cfg.get("background_ratio", 0.10),
            random_seed=cfg.get("random_seed", 42),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_source_tag(tile_path: Path) -> str:
    """Read the ``source_tif`` TIFF tag written by :mod:`src.data.tiler`.

    Falls back to the full tile stem (e.g. ``scene_001_320_640``) when the
    tag is absent, which is safe for single-image datasets.

    Args:
        tile_path: Path to a GeoTIFF tile.

    Returns:
        The ``source_tif`` tag value (typically the basename of the original
        ``.tif``, e.g. ``"scene_001.tif"``), or the tile stem on failure.
    """
    try:
        with rasterio.open(tile_path) as ds:
            return ds.tags().get("source_tif", tile_path.stem)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cannot read TIFF tags from '%s': %s", tile_path.name, exc)
        return tile_path.stem


def _is_background(label_path: Path) -> bool:
    """Return ``True`` if a YOLO label file is empty (background tile).

    An empty label file or a file containing only whitespace is treated as
    a background tile.  A missing file is also treated as background (the
    annotator silently skipped it).

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        ``True`` if the file is absent or contains no annotation lines.
    """
    if not label_path.exists():
        return True
    return label_path.read_text().strip() == ""


def _assign_splits(
    source_images: List[str],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, str]:
    """Assign each source image name to a split string.

    Images are assigned in order after shuffling.  The split boundaries are
    computed as integer indices to avoid floating-point rounding issues.

    Args:
        source_images: Unique source image names (already shuffled).
        train_ratio: Fraction for the training set.
        val_ratio: Fraction for the validation set.

    Returns:
        Mapping ``{source_image_name: split_name}`` where ``split_name``
        is one of ``"train"``, ``"val"``, or ``"test"``.

    Note:
        When there are fewer source images than splits (e.g., 2 images for
        a 3-way split), some splits will be empty.  A warning is logged.
    """
    n = len(source_images)
    n_train = max(1, round(n * train_ratio))
    n_val   = max(0, round(n * val_ratio))
    # test gets the remainder — no rounding error accumulation.
    n_test  = n - n_train - n_val

    if n_test < 0:
        # Clamping: val takes from train if not enough images.
        n_val  += n_test
        n_test  = 0

    if n_val == 0:
        logger.warning(
            "Not enough source images (%d) to populate the validation split. "
            "Consider adding more source images or reducing val_ratio.",
            n,
        )
    if n_test == 0:
        logger.warning(
            "Not enough source images (%d) to populate the test split. "
            "Consider adding more source images or reducing test_ratio.",
            n,
        )

    assignment: Dict[str, str] = {}
    for i, name in enumerate(source_images):
        if i < n_train:
            assignment[name] = "train"
        elif i < n_train + n_val:
            assignment[name] = "val"
        else:
            assignment[name] = "test"

    return assignment


def _move_pair(
    src_img: Path,
    src_lbl: Optional[Path],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> None:
    """Atomically move an image tile and its label file to destination directories.

    Args:
        src_img: Source image path (GeoTIFF tile).
        src_lbl: Source label path (``.txt``), or ``None`` if absent.
        dst_img_dir: Destination directory for the image.
        dst_lbl_dir: Destination directory for the label.
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.move(str(src_img), str(dst_img_dir / src_img.name))
    logger.debug("  MOVE  %s  →  %s/", src_img.name, dst_img_dir)

    if src_lbl is not None and src_lbl.exists():
        shutil.move(str(src_lbl), str(dst_lbl_dir / src_lbl.name))


# ---------------------------------------------------------------------------
# Public API — splitting
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """Assigns GeoTIFF tiles to train / val / test partitions.

    Grouping is performed at the *source-image level* to guarantee zero
    spatial leakage: tiles that share an origin image always end up in the
    same partition.  See module docstring for motivation.

    Args:
        config: :class:`SplitConfig` specifying ratios and random seed.
    """

    def __init__(self, config: SplitConfig) -> None:
        self._cfg = config

    def split(
        self,
        tiles_dir: Path,
        labels_raw_dir: Path,
        images_dir: Path,
        labels_dir: Path,
    ) -> Dict[str, List[Path]]:
        """Distribute tiles and labels into split subdirectories.

        The source image for each tile is identified by the ``source_tif``
        TIFF tag.  All tiles sharing a source are assigned to the same
        split, then physically moved from *tiles_dir* / *labels_raw_dir*
        into *images_dir/<split>/* and *labels_dir/<split>/*.

        Args:
            tiles_dir: Directory of GeoTIFF tiles (step 1 output).
            labels_raw_dir: Directory of YOLO ``.txt`` files (step 2 output).
            images_dir: Root output directory; subdirectories
                ``train/``, ``val/``, ``test/`` are created automatically.
            labels_dir: Root output directory for labels; mirrored
                structure to *images_dir*.

        Returns:
            Mapping ``{split_name: [tile_path, ...]}`` of tiles moved into
            each split.

        Raises:
            FileNotFoundError: If *tiles_dir* does not exist.
            RuntimeError: If no ``.tif`` tiles are found.
        """
        if not tiles_dir.exists():
            raise FileNotFoundError(f"tiles_dir does not exist: {tiles_dir}")

        tile_paths = sorted(tiles_dir.glob("*.tif"))
        if not tile_paths:
            raise RuntimeError(f"No .tif tiles found in '{tiles_dir}'.")

        # ------------------------------------------------------------------
        # 1. Group tiles by source image.
        # ------------------------------------------------------------------
        source_to_tiles: Dict[str, List[Path]] = {}
        for tp in tile_paths:
            src = _read_source_tag(tp)
            source_to_tiles.setdefault(src, []).append(tp)

        source_images = sorted(source_to_tiles.keys())
        logger.info(
            "Split: %d tile(s) from %d source image(s). "
            "Ratios: train=%.2f  val=%.2f  test=%.2f",
            len(tile_paths), len(source_images),
            self._cfg.train_ratio, self._cfg.val_ratio, self._cfg.test_ratio,
        )

        # ------------------------------------------------------------------
        # 2. Shuffle source images and assign to splits.
        # ------------------------------------------------------------------
        rng = random.Random(self._cfg.random_seed)
        rng.shuffle(source_images)

        assignment = _assign_splits(
            source_images, self._cfg.train_ratio, self._cfg.val_ratio
        )

        split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        result: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

        # ------------------------------------------------------------------
        # 3. Move tiles and labels to split directories.
        # ------------------------------------------------------------------
        for src_name, tiles in source_to_tiles.items():
            split_name = assignment.get(src_name, "train")
            dst_img    = images_dir / split_name
            dst_lbl    = labels_dir / split_name

            for tile_path in tiles:
                label_path = labels_raw_dir / f"{tile_path.stem}.txt"
                _move_pair(
                    tile_path, label_path if label_path.exists() else None,
                    dst_img, dst_lbl,
                )
                result[split_name].append(dst_img / tile_path.name)
                split_counts[split_name] += 1

        for split, count in split_counts.items():
            logger.info("  %-5s : %d tile(s)", split, count)

        return result


# ---------------------------------------------------------------------------
# Public API — background balancing
# ---------------------------------------------------------------------------

class BackgroundBalancer:
    """Reduces background-tile over-representation within dataset splits.

    Excess background tiles are *moved* to an archive directory rather
    than deleted, preserving the option to restore them.

    The target count of background tiles is derived from the number of
    annotated tiles via::

        target_bg = annotated × (ratio / (1 − ratio))

    Args:
        config: :class:`BalanceConfig` specifying the target ratio and seed.
    """

    def __init__(self, config: BalanceConfig) -> None:
        self._cfg = config

    def balance(
        self,
        images_dir: Path,
        labels_dir: Path,
        archive_dir: Path,
        splits: List[str],
    ) -> None:
        """Balance background tiles for the specified splits.

        For each split listed in *splits*, background tiles in excess of
        the ``background_ratio`` target are moved to::

            <archive_dir>/images/<split>/
            <archive_dir>/labels/<split>/

        Args:
            images_dir: Root directory containing ``<split>/`` subdirectories
                with GeoTIFF tiles.
            labels_dir: Root directory containing ``<split>/`` subdirectories
                with YOLO ``.txt`` label files.
            archive_dir: Root directory for archived excess tiles.
            splits: List of split names to process (e.g., ``["train", "val"]``).
        """
        ratio = self._cfg.background_ratio
        rng   = random.Random(self._cfg.random_seed)

        logger.info(
            "Background balancing: target ratio=%.2f  splits=%s",
            ratio, splits,
        )

        for split in splits:
            self._balance_split(split, images_dir, labels_dir, archive_dir, ratio, rng)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _balance_split(
        self,
        split: str,
        images_dir: Path,
        labels_dir: Path,
        archive_dir: Path,
        ratio: float,
        rng: random.Random,
    ) -> None:
        """Balance one split.

        Args:
            split: Split name (e.g., ``"train"``).
            images_dir: Root images directory (parent of ``<split>/``).
            labels_dir: Root labels directory (parent of ``<split>/``).
            archive_dir: Root archive directory.
            ratio: Target background fraction.
            rng: Seeded :class:`random.Random` for reproducibility.
        """
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split

        if not img_split_dir.exists():
            logger.warning(
                "Balance: images/%s/ does not exist — skipping.", split
            )
            return

        # Categorise tiles.
        annotated: List[str]  = []
        background: List[str] = []

        for img_path in sorted(img_split_dir.glob("*.tif")):
            stem       = img_path.stem
            label_path = lbl_split_dir / f"{stem}.txt"
            if _is_background(label_path):
                background.append(stem)
            else:
                annotated.append(stem)

        n_ann = len(annotated)
        n_bg  = len(background)

        if n_ann == 0:
            logger.warning(
                "Balance: no annotated tiles in '%s' — skipping to avoid "
                "removing all data.",
                split,
            )
            return

        # Desired background count: bg = ann × ratio / (1 − ratio)
        target_bg = int(n_ann * ratio / (1.0 - ratio))
        target_bg = min(target_bg, n_bg)  # cannot keep more than we have

        rng.shuffle(background)
        keep_bg = background[:target_bg]
        move_bg = background[target_bg:]

        n_total_new    = n_ann + target_bg
        actual_bg_pct  = target_bg / n_total_new * 100.0 if n_total_new > 0 else 0.0

        logger.info(
            "  [%s]  annotated=%d  bg_before=%d  bg_kept=%d (%.1f%%)  bg_archived=%d"
            "  new_total=%d",
            split.upper(), n_ann, n_bg, target_bg, actual_bg_pct,
            len(move_bg), n_total_new,
        )

        if not move_bg:
            return

        arch_img = archive_dir / "images" / split
        arch_lbl = archive_dir / "labels" / split
        arch_img.mkdir(parents=True, exist_ok=True)
        arch_lbl.mkdir(parents=True, exist_ok=True)

        for stem in move_bg:
            src_img = img_split_dir / f"{stem}.tif"
            src_lbl = lbl_split_dir / f"{stem}.txt"

            if src_img.exists():
                shutil.move(str(src_img), str(arch_img / f"{stem}.tif"))
            if src_lbl.exists():
                shutil.move(str(src_lbl), str(arch_lbl / f"{stem}.txt"))

        logger.info(
            "  [%s]  Archived %d background tile(s) to '%s'.",
            split.upper(), len(move_bg), archive_dir,
        )
