"""
src/vessels_detect/preprocessing/steps/tiling.py
--------------------------------------------------
Stage 5 — Raw GeoTIFF tiling with YOLO OBB label projection.

This step is intentionally **radiometric-free**:

- Zero contrast stretching.
- Zero gamma correction.
- Original source dtype and band count preserved verbatim.
- Every tile is written — no blank/uniform-tile filtering.

Each output tile is a self-contained, valid GeoTIFF that embeds:

- Its own Coordinate Reference System (CRS), copied from the source.
- Its own Affine transform (pixel-space → map-space), anchored at the
  tile's top-left corner.
- Provenance TIFF tags: source filename, pixel offsets, original image
  dimensions, and tile size.

For every image tile the step also generates the matching YOLO OBB label
file by projecting all annotations from the source label into tile-relative
normalised coordinates.  Annotations whose visible area inside the tile is
smaller than ``min_visible_frac`` are silently discarded.

Configuration path in the YAML (``cfg["tiling"]``)::

    tiling:
      splits:           [train, val]   # which splits to tile
      tile_size:        640            # output tile height and width (px)
      overlap:          64             # pixel overlap between adjacent tiles
      compress:         lzw            # GeoTIFF compression codec
      min_visible_frac: 0.10           # min OBB visible fraction to keep label
      images_subdir:    images         # subfolder name for tile GeoTIFFs
      labels_subdir:    labels         # subfolder name for tile label files

Input layout (produced by Stage 4 — split)::

    dataset_dir/
      images/
        {split}/   *.tif
      labels/
        {split}/   *.txt   (YOLO OBB: class x1 y1 x2 y2 x3 y3 x4 y4, normalised)

Output layout::

    tiled_dir/
      {images_subdir}/
        {split}/   {stem}_{x_off}_{y_off}.tif
      {labels_subdir}/
        {split}/   {stem}_{x_off}_{y_off}.txt

Typical standalone usage::

    from pathlib import Path
    from src.vessels_detect.preprocessing.steps.tiling import TilingStep

    step = TilingStep()
    step.run(cfg)          # cfg is the fully-resolved config dict from manager.py
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon

from src.vessels_detect.preprocessing.steps.base import BaseStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TilingStepConfig:
    """All hyperparameters consumed by :class:`TilingStep`.

    Attributes:
        splits: Which dataset splits to tile.  Must be a subset of the
            split names produced by Stage 4 (``"train"``, ``"val"``,
            ``"test"``).
        tile_size: Output tile width and height in pixels.  Edge tiles are
            zero-padded (or nodata-padded when the source defines nodata) to
            this exact size so that every tile has uniform dimensions.
        overlap: Pixel overlap between adjacent tiles in both axes.
            Effective stride = ``tile_size - overlap``.
        compress: GeoTIFF compression codec.  ``"lzw"`` (lossless) is the
            recommended default for multi-dtype imagery.
        min_visible_frac: Minimum fraction of an OBB's area that must fall
            inside the tile for the corresponding label line to be written.
            Set to ``0.0`` to keep all annotations touching the tile.
        images_subdir: Subdirectory name used for tile GeoTIFFs inside
            ``tiled_dir``.  The full output path for a split is
            ``tiled_dir / images_subdir / split /``.
        labels_subdir: Subdirectory name used for tile label files inside
            ``tiled_dir``.  The full output path for a split is
            ``tiled_dir / labels_subdir / split /``.
    """

    splits: List[str] = field(default_factory=lambda: ["train", "val", "test"])
    tile_size: int = 640
    overlap: int = 64
    compress: str = "lzw"
    min_visible_frac: float = 0.10
    images_subdir: str = "images"
    labels_subdir: str = "labels"

    def __post_init__(self) -> None:
        if self.overlap >= self.tile_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be strictly less than "
                f"tile_size ({self.tile_size})."
            )
        if self.min_visible_frac < 0.0 or self.min_visible_frac > 1.0:
            raise ValueError(
                f"min_visible_frac must be in [0.0, 1.0], "
                f"got {self.min_visible_frac}."
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "TilingStepConfig":
        """Construct from the ``cfg["tiling"]`` sub-dictionary.

        Args:
            cfg: The ``tiling`` section parsed from the YAML config.
                Unknown keys are silently ignored.

        Returns:
            A fully populated :class:`TilingStepConfig` instance.
        """
        splits_raw = cfg.get("splits", ["train", "val", "test"])
        return cls(
            splits=list(splits_raw),
            tile_size=int(cfg.get("tile_size", 640)),
            overlap=int(cfg.get("overlap", 64)),
            compress=str(cfg.get("compress", "lzw")),
            min_visible_frac=float(cfg.get("min_visible_frac", 0.10)),
            images_subdir=str(cfg.get("images_subdir", "images")),
            labels_subdir=str(cfg.get("labels_subdir", "labels")),
        )


# ---------------------------------------------------------------------------
# GeoTIFF helpers — pure tiling, no radiometric processing
# ---------------------------------------------------------------------------

def _pad_tile(
    arr: np.ndarray,
    tile_size: int,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Zero-pad (or nodata-pad) ``(C, H, W)`` to ``(C, tile_size, tile_size)``.

    Args:
        arr: Source array with shape ``(C, h, w)`` where both
            ``h ≤ tile_size`` and ``w ≤ tile_size``.
        tile_size: Target spatial dimension.
        fill_value: Value used to fill the padded region.  Should be the
            source nodata value when available; defaults to ``0``.

    Returns:
        Array of shape ``(C, tile_size, tile_size)`` in the source dtype.
    """
    pad_h = tile_size - arr.shape[1]
    pad_w = tile_size - arr.shape[2]
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(
        arr,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=fill_value,
    )


def _build_tile_profile(
    src: rasterio.DatasetReader,
    tile_transform: Affine,
    tile_size: int,
    compress: str,
) -> dict:
    """Construct a rasterio write profile that preserves source dtype and CRS.

    Unlike the original :func:`tiler._build_output_profile`, this variant
    never forces ``uint8`` — the source dtype is carried through verbatim.

    Args:
        src: Open source rasterio dataset.
        tile_transform: Affine transform anchored at the tile's top-left
            corner in the source CRS.
        tile_size: Tile height and width in pixels.
        compress: GeoTIFF compression codec.

    Returns:
        Profile dictionary suitable for :func:`rasterio.open`.
    """
    return {
        "driver":     "GTiff",
        "dtype":      src.dtypes[0],       # ← preserve source dtype
        "count":      src.count,            # ← preserve band count
        "width":      tile_size,
        "height":     tile_size,
        "crs":        src.crs,
        "transform":  tile_transform,
        "compress":   compress,
        "predictor":  2,                    # horizontal differencing (safe for all dtypes)
        "tiled":      True,
        "blockxsize": min(256, tile_size),
        "blockysize": min(256, tile_size),
    }


def tile_image_raw(
    tif_path: Path,
    output_image_dir: Path,
    tile_size: int,
    overlap: int,
    compress: str,
) -> List[Tuple[Path, int, int]]:
    """Tile a GeoTIFF into fixed-size patches with no radiometric processing.

    Every pixel value is preserved exactly as read from *tif_path*.  No
    stretching, normalisation, or gamma curve is applied.  Edge tiles are
    padded to ``tile_size`` using the source nodata value (or 0 when nodata
    is not defined).

    All tiles are written regardless of content (no blank-tile filtering).

    Args:
        tif_path: Path to the source GeoTIFF.
        output_image_dir: Directory where tile GeoTIFFs will be written.
        tile_size: Output tile height and width in pixels.
        overlap: Pixel overlap between adjacent tiles.  Effective stride =
            ``tile_size - overlap``.
        compress: GeoTIFF compression codec passed to rasterio.

    Returns:
        List of ``(tile_path, x_off, y_off)`` tuples — one entry per tile
        written.  The pixel offsets are relative to the source image origin
        and can be used to correlate tiles with their labels.
    """
    stride = tile_size - overlap

    tile_records: List[Tuple[Path, int, int]] = []
    stem = tif_path.stem

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        nodata = src.nodata
        fill_value = float(nodata) if nodata is not None else 0.0

        n_cols = math.ceil(W / stride)
        n_rows = math.ceil(H / stride)

        logger.debug(
            "  Tiling %s (%d×%d px, %d bands, dtype=%s) → %d×%d grid",
            tif_path.name, W, H, src.count, src.dtypes[0], n_cols, n_rows,
        )

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                x_off = col_idx * stride
                y_off = row_idx * stride

                # Actual window extent clipped to image bounds.
                win_w = min(tile_size, W - x_off)
                win_h = min(tile_size, H - y_off)

                window = Window(x_off, y_off, win_w, win_h)
                raw_tile = src.read(window=window)          # shape: (C, win_h, win_w)

                # Pad edge tiles to uniform tile_size × tile_size.
                padded = _pad_tile(raw_tile, tile_size, fill_value)

                tile_transform = src.window_transform(window)
                profile = _build_tile_profile(src, tile_transform, tile_size, compress)

                out_path = output_image_dir / f"{stem}_{x_off}_{y_off}.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(padded)
                    dst.update_tags(
                        source_tif=tif_path.name,
                        col_off=str(x_off),
                        row_off=str(y_off),
                        src_width=str(W),
                        src_height=str(H),
                        tile_size=str(tile_size),
                    )

                tile_records.append((out_path, x_off, y_off))

    return tile_records


# ---------------------------------------------------------------------------
# YOLO OBB label projection
# ---------------------------------------------------------------------------

def _parse_yolo_obb_line(line: str) -> Optional[Tuple[int, np.ndarray]]:
    """Parse a single YOLO OBB label line.

    Expected format::

        class_id x1 y1 x2 y2 x3 y3 x4 y4

    where all coordinates are normalised to [0, 1] relative to the image.

    Args:
        line: Raw text line (may include trailing newline / spaces).

    Returns:
        ``(class_id, corners)`` where ``corners`` is shape ``(4, 2)`` in
        ``[[x, y], ...]`` order.  Returns ``None`` for blank or malformed
        lines.
    """
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        cls_id = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float64).reshape(4, 2)  # (4, 2) x/y pairs
    except ValueError:
        return None
    return cls_id, coords


def _obb_visible_fraction(
    corners_px: np.ndarray,
    tile_polygon: Polygon,
) -> float:
    """Compute the fraction of an OBB's area that falls inside the tile.

    Args:
        corners_px: OBB corner coordinates in pixel space, shape ``(4, 2)``.
        tile_polygon: Shapely polygon representing the tile rectangle.

    Returns:
        Visible fraction in ``[0, 1]``.  Returns ``0.0`` if the OBB has
        zero area (degenerate polygon).
    """
    obb_polygon = Polygon(corners_px)
    obb_area = obb_polygon.area
    if obb_area < 1e-9:
        return 0.0
    intersection = obb_polygon.intersection(tile_polygon)
    return intersection.area / obb_area


def project_labels_to_tile(
    label_path: Optional[Path],
    img_width: int,
    img_height: int,
    x_off: int,
    y_off: int,
    tile_size: int,
    min_visible_frac: float,
) -> List[str]:
    """Project YOLO OBB annotations from image space into tile space.

    Annotations whose visible area inside the tile is smaller than
    ``min_visible_frac`` are discarded.  Surviving annotations are
    re-expressed as normalised tile-relative coordinates (corners that
    fall outside the tile may have values outside ``[0, 1]``, which is
    valid for YOLO OBB — the model is trained on full-box coordinates).

    Args:
        label_path: Path to the YOLO OBB label file, or ``None`` / missing
            file (results in an empty label file for the tile).
        img_width: Width of the source image in pixels.
        img_height: Height of the source image in pixels.
        x_off: Horizontal pixel offset of the tile's top-left corner in
            the source image.
        y_off: Vertical pixel offset of the tile's top-left corner in the
            source image.
        tile_size: Tile width and height in pixels (padded dimensions).
        min_visible_frac: Minimum visible fraction threshold.

    Returns:
        List of YOLO OBB label lines ready to be joined with ``\\n`` and
        written to a ``.txt`` file.  Empty list when there are no
        qualifying annotations.
    """
    if label_path is None or not label_path.exists():
        return []

    lines = label_path.read_text(encoding="utf-8").splitlines()

    # Pre-compute the tile polygon once in pixel space.
    tile_polygon = Polygon([
        (x_off,             y_off),
        (x_off + tile_size, y_off),
        (x_off + tile_size, y_off + tile_size),
        (x_off,             y_off + tile_size),
    ])

    output_lines: List[str] = []

    for raw_line in lines:
        parsed = _parse_yolo_obb_line(raw_line)
        if parsed is None:
            continue

        cls_id, corners_norm = parsed   # corners_norm: (4, 2) normalised

        # ── Convert to pixel coordinates in source image space ──────────
        corners_px = corners_norm * np.array([[img_width, img_height]])   # (4, 2)

        # ── Visibility check ─────────────────────────────────────────────
        vis = _obb_visible_fraction(corners_px, tile_polygon)
        if vis < min_visible_frac:
            continue

        # ── Re-normalise to tile space ───────────────────────────────────
        # Subtract the tile origin, then normalise by tile_size (the padded
        # tile dimension). Coordinates outside [0, 1] are intentional and
        # valid for partial boxes at tile edges.
        corners_tile = (corners_px - np.array([[x_off, y_off]])) / tile_size  # (4, 2)

        # Flatten to: class_id x1 y1 x2 y2 x3 y3 x4 y4
        flat = corners_tile.flatten()
        parts = [str(cls_id)] + [f"{v:.6f}" for v in flat]
        output_lines.append(" ".join(parts))

    return output_lines


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

class TilingStep(BaseStep):
    """Stage 5 — raw GeoTIFF tiling with YOLO OBB label projection.

    Reads the split dataset produced by :class:`SplitStep` and tiles every
    selected split's images into fixed-size GeoTIFF patches, generating
    matching YOLO OBB label files for each tile.

    No radiometric processing is applied.  Source dtype, band count, CRS,
    and raw pixel values are all preserved verbatim.

    Input layout (Stage 4 output)::

        dataset_dir/
          images/{split}/   *.tif
          labels/{split}/   *.txt

    Output layout::

        tiled_dir/
          {images_subdir}/{split}/   {stem}_{x_off}_{y_off}.tif
          {labels_subdir}/{split}/   {stem}_{x_off}_{y_off}.txt

    The ``splits``, ``images_subdir``, and ``labels_subdir`` config keys
    control which splits are processed and where their outputs land, making
    it straightforward to tile only the training set or to redirect outputs
    without touching the code.
    """

    NAME = "tiling"

    # ------------------------------------------------------------------
    # BaseStep interface
    # ------------------------------------------------------------------

    def run(self, cfg: dict) -> None:
        """Execute the tiling step.

        Reads the split dataset produced by Stage 4 and tiles every selected
        split's images into fixed-size GeoTIFF patches, generating matching
        YOLO OBB label files for each tile.  No radiometric processing is
        applied at any point.

        Args:
            cfg: Fully resolved configuration dictionary from
                :func:`~manager.load_config`.  Expected keys:

                ``cfg["paths"]["dataset_dir"]``
                    Root of the split dataset (output of Stage 4).
                    Expected sub-layout: ``images/{split}/`` and
                    ``labels/{split}/``.

                ``cfg["paths"]["tiled_dir"]``
                    Root directory where tiled outputs are written.
                    Sub-layout: ``{images_subdir}/{split}/`` and
                    ``{labels_subdir}/{split}/``.

                ``cfg["tiling"]``
                    Step-specific hyperparameters; see
                    :class:`TilingStepConfig`.

        Raises:
            KeyError: If required config keys are absent.
            FileNotFoundError: If a configured split directory does not exist.
            ValueError: If ``TilingStepConfig`` validation fails.
        """
        step_cfg = TilingStepConfig.from_dict(cfg.get("tiling", {}))
        dataset_dir: Path = cfg["paths"]["dataset_dir"]
        tiled_dir: Path   = cfg["paths"]["tiled_dir"]

        logger.info("Tiling step configuration:")
        logger.info("  dataset_dir      : %s", dataset_dir)
        logger.info("  tiled_dir        : %s", tiled_dir)
        logger.info("  splits           : %s", step_cfg.splits)
        logger.info("  tile_size        : %d px", step_cfg.tile_size)
        logger.info("  overlap          : %d px", step_cfg.overlap)
        logger.info("  stride           : %d px", step_cfg.tile_size - step_cfg.overlap)
        logger.info("  min_visible_frac : %.2f", step_cfg.min_visible_frac)
        logger.info("  compress         : %s", step_cfg.compress)
        logger.info("  images_subdir    : %s", step_cfg.images_subdir)
        logger.info("  labels_subdir    : %s", step_cfg.labels_subdir)

        total_tiles  = 0
        total_labels = 0

        for split in step_cfg.splits:
            # Stage 4 (split.py) writes:  dataset_dir/images/{split}/
            #                             dataset_dir/labels/{split}/
            split_image_dir = dataset_dir / "images" / split
            split_label_dir = dataset_dir / "labels" / split

            if not split_image_dir.exists():
                raise FileNotFoundError(
                    f"Split image directory not found: {split_image_dir}. "
                    f"Ensure Stage 4 (split) has been run before Stage 5 (tiling)."
                )

            tif_files = sorted(split_image_dir.glob("*.tif"))
            if not tif_files:
                logger.warning(
                    "No .tif files found in '%s' — skipping split.", split_image_dir
                )
                continue

            # Output layout: tiled_dir/{images_subdir}/{split}/
            #                tiled_dir/{labels_subdir}/{split}/
            out_image_dir = tiled_dir / step_cfg.images_subdir / split
            out_label_dir = tiled_dir / step_cfg.labels_subdir / split
            out_image_dir.mkdir(parents=True, exist_ok=True)
            out_label_dir.mkdir(parents=True, exist_ok=True)

            logger.info("")
            logger.info("Processing split '%s' (%d image(s))", split, len(tif_files))

            split_tiles  = 0
            split_labels = 0

            for tif_path in tif_files:
                n_tiles, n_labels = self._tile_single_image(
                    tif_path=tif_path,
                    label_dir=split_label_dir,
                    out_image_dir=out_image_dir,
                    out_label_dir=out_label_dir,
                    step_cfg=step_cfg,
                )
                split_tiles  += n_tiles
                split_labels += n_labels
                logger.info(
                    "  %-40s → %d tile(s), %d label line(s)",
                    tif_path.name, n_tiles, n_labels,
                )

            logger.info(
                "  Split '%s' complete: %d tile(s), %d total label line(s).",
                split, split_tiles, split_labels,
            )
            total_tiles  += split_tiles
            total_labels += split_labels

        logger.info("")
        logger.info(
            "Tiling complete.  Total tiles: %d  |  Total label lines: %d",
            total_tiles, total_labels,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tile_single_image(
        tif_path: Path,
        label_dir: Path,
        out_image_dir: Path,
        out_label_dir: Path,
        step_cfg: TilingStepConfig,
    ) -> Tuple[int, int]:
        """Tile one GeoTIFF and project its labels onto every tile.

        Args:
            tif_path: Source GeoTIFF path.
            label_dir: Directory containing YOLO OBB label files
                (stem-matched, ``.txt``).
            out_image_dir: Destination directory for tile GeoTIFFs.
            out_label_dir: Destination directory for tile label files.
            step_cfg: Resolved tiling hyperparameters.

        Returns:
            Tuple of ``(n_tiles_written, n_label_lines_written)``.
        """
        label_path: Optional[Path] = label_dir / f"{tif_path.stem}.txt"
        if not label_path.exists():
            label_path = None
            logger.debug("  No label file found for %s — tiles will be empty.", tif_path.stem)

        # Retrieve image dimensions without loading the full raster.
        with rasterio.open(tif_path) as src:
            img_w, img_h = src.width, src.height

        tile_records = tile_image_raw(
            tif_path=tif_path,
            output_image_dir=out_image_dir,
            tile_size=step_cfg.tile_size,
            overlap=step_cfg.overlap,
            compress=step_cfg.compress,
        )

        total_label_lines = 0

        for _tile_path, x_off, y_off in tile_records:
            label_lines = project_labels_to_tile(
                label_path=label_path,
                img_width=img_w,
                img_height=img_h,
                x_off=x_off,
                y_off=y_off,
                tile_size=step_cfg.tile_size,
                min_visible_frac=step_cfg.min_visible_frac,
            )
            stem_tile = _tile_path.stem
            out_txt = out_label_dir / f"{stem_tile}.txt"
            out_txt.write_text("\n".join(label_lines), encoding="utf-8")
            total_label_lines += len(label_lines)

        return len(tile_records), total_label_lines