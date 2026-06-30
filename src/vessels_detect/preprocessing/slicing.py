"""
src/vessels_detect/preprocessing/slicing.py
----------------------------------------------
Step 4 - Slicing (tiling) + label projection.

Cuts one split image into fixed-size GeoTIFF tiles using rasterio windowed
I/O (one tile's pixels in memory at a time - never the full image), and
projects the image-level YOLO OBB labels into tile-relative normalised
coordinates for every tile produced.

No radiometric processing happens here: source dtype, band count, and pixel
values are carried through verbatim from the (already-enhanced) input.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tile writing
# ---------------------------------------------------------------------------

def _pad_tile(arr: np.ndarray, tile_size: int, fill_value: float) -> np.ndarray:
    """Pad ``(C, h, w)`` up to ``(C, tile_size, tile_size)``."""
    pad_h, pad_w = tile_size - arr.shape[1], tile_size - arr.shape[2]
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=fill_value)


def _write_tiles(
    src_path: Path,
    out_image_dir: Path,
    tile_size: int,
    overlap: int,
    compress: str,
) -> List[Tuple[Path, int, int]]:
    """Stream-read one window at a time and write each tile immediately.

    Memory footprint is one ``tile_size x tile_size`` array, regardless of
    the source GeoTIFF's full resolution.
    """
    stride = tile_size - overlap
    stem = src_path.stem
    tile_records: List[Tuple[Path, int, int]] = []

    with rasterio.open(src_path) as src:
        W, H = src.width, src.height
        fill_value = float(src.nodata) if src.nodata is not None else 0.0
        n_cols, n_rows = math.ceil(W / stride), math.ceil(H / stride)

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                x_off, y_off = col_idx * stride, row_idx * stride
                win_w, win_h = min(tile_size, W - x_off), min(tile_size, H - y_off)

                window = Window(x_off, y_off, win_w, win_h)
                tile = _pad_tile(src.read(window=window), tile_size, fill_value)

                # Skip uniform (no-content) tiles.
                if tile.min() == tile.max():
                    continue

                tile_transform = src.window_transform(window)
                profile = {
                    "driver": "GTiff", "dtype": src.dtypes[0], "count": src.count,
                    "width": tile_size, "height": tile_size, "crs": src.crs,
                    "transform": tile_transform, "compress": compress, "predictor": 2,
                    "tiled": True, "blockxsize": min(256, tile_size), "blockysize": min(256, tile_size),
                }

                out_path = out_image_dir / f"{stem}_{x_off}_{y_off}.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(tile)
                    dst.update_tags(source_tif=src_path.name, col_off=str(x_off), row_off=str(y_off))

                tile_records.append((out_path, x_off, y_off))

    return tile_records


# ---------------------------------------------------------------------------
# Label projection
# ---------------------------------------------------------------------------

def _parse_yolo_obb_line(line: str) -> Optional[Tuple[int, np.ndarray]]:
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        return int(parts[0]), np.array(parts[1:], dtype=np.float64).reshape(4, 2)
    except ValueError:
        return None


def _obb_visible_fraction(corners_px: np.ndarray, tile_polygon: Polygon) -> float:
    obb_polygon = Polygon(corners_px)
    if obb_polygon.area < 1e-9:
        return 0.0
    return obb_polygon.intersection(tile_polygon).area / obb_polygon.area


def _project_labels_to_tile(
    label_path: Optional[Path],
    img_width: int,
    img_height: int,
    x_off: int,
    y_off: int,
    tile_size: int,
    min_visible_frac: float,
) -> List[str]:
    """Re-express image-level normalised OBBs as tile-relative normalised OBBs."""
    if label_path is None or not label_path.exists():
        return []

    tile_polygon = Polygon([
        (x_off, y_off), (x_off + tile_size, y_off),
        (x_off + tile_size, y_off + tile_size), (x_off, y_off + tile_size),
    ])

    output_lines: List[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_yolo_obb_line(raw_line)
        if parsed is None:
            continue
        cls_id, corners_norm = parsed

        corners_px = corners_norm * np.array([[img_width, img_height]])
        if _obb_visible_fraction(corners_px, tile_polygon) < min_visible_frac:
            continue

        corners_tile = (corners_px - np.array([[x_off, y_off]])) / tile_size
        corners_tile = np.clip(corners_tile, 0.0, 1.0)

        flat = corners_tile.flatten()
        output_lines.append(" ".join([str(cls_id)] + [f"{v:.6f}" for v in flat]))

    return output_lines


# ---------------------------------------------------------------------------
# Public entry point - used by scripts/preprocessing.py
# ---------------------------------------------------------------------------

def slice_image(
    image_path: Path,
    label_path: Optional[Path],
    out_dir: Path,
    split: str,
    cfg: Dict,
) -> Tuple[int, int]:
    """Tile one split image and project its labels onto every tile produced.

    Args:
        image_path: Path to the split-stage image (e.g.
            ``dataset_dir/images/train/scene_001.tif``).
        label_path: Path to the matching image-level YOLO label file, or
            ``None`` if it doesn't exist (tiles will get empty labels).
        out_dir: Root tiled-output directory (``cfg["paths"]["tiled_dir"]``).
        split: Split name (``"train"``, ``"val"``, or ``"test"``) - used to
            build the output sub-directory.
        cfg: Full resolved pipeline config (uses ``cfg["tiling"]``).

    Returns:
        Tuple of ``(n_tiles_written, n_label_lines_written)``.
    """
    params = cfg["tiling"]
    tile_size = params.get("tile_size", 640)
    overlap = params.get("overlap", 0)
    compress = params.get("compress", "lzw")
    min_visible_frac = params.get("min_visible_frac", 0.10)
    images_subdir = params.get("images_subdir", "images")
    labels_subdir = params.get("labels_subdir", "labels")

    out_image_dir = out_dir / images_subdir / split
    out_label_dir = out_dir / labels_subdir / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(image_path) as src:
        img_w, img_h = src.width, src.height

    tile_records = _write_tiles(image_path, out_image_dir, tile_size, overlap, compress)

    total_label_lines = 0
    for tile_path, x_off, y_off in tile_records:
        lines = _project_labels_to_tile(
            label_path, img_w, img_h, x_off, y_off, tile_size, min_visible_frac
        )
        (out_label_dir / f"{tile_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        total_label_lines += len(lines)

    logger.info(
        "  [slicing] %s -> %d tile(s), %d label line(s)",
        image_path.name, len(tile_records), total_label_lines,
    )
    return len(tile_records), total_label_lines
