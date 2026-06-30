"""
src/vessels_detect/preprocessing/data_augmentation.py
---------------------------------------------------------
Step 6 - Data augmentation (final step, tile-level).

For every tile in the configured split(s), generates a fixed number of
pixel-augmented copies alongside the original. Augmentations here are
purely photometric (noise, blur, brightness, ...) so the YOLO OBB label
file is copied through unchanged for every augmented copy - the boxes
themselves don't move.

Runs last in the pipeline, after background filtering, so augmented copies
are only generated from the final, ratio-balanced tile set (otherwise the
background ratio computed in Step 5 would no longer hold).

Memory safety
~~~~~~~~~~~~~
Tiles are processed one at a time - only a single tile's pixel array is
ever held in memory, regardless of how many tiles or augmentations are
requested.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List

import albumentations as A
import numpy as np
import rasterio

logger = logging.getLogger(__name__)

_IMAGE_GLOBS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def _round_block_size(dim: int) -> int:
    """Round a dimension down to the nearest multiple of 16, GDAL's minimum
    tile-size granularity, with a floor of 16."""
    return max(16, (dim // 16) * 16)


# ---------------------------------------------------------------------------
# Augmentation pipeline builder
# ---------------------------------------------------------------------------

def _build_transform(params: Dict) -> A.Compose:
    """Build the Albumentations pipeline from config.

    Only photometric transforms are supported here (label coordinates are
    not touched). Each entry in ``cfg["data_augmentation"]["transforms"]``
    maps an Albumentations class name to its constructor kwargs.
    """
    transform_specs = params.get("transforms", [{"name": "GaussNoise", "p": 1.0, "std_range": [0.0, 0.04]}])

    ops = []
    for spec in transform_specs:
        spec = dict(spec)
        name = spec.pop("name")
        cls = getattr(A, name)
        ops.append(cls(**spec))

    return A.Compose(ops)


# ---------------------------------------------------------------------------
# Per-tile augmentation
# ---------------------------------------------------------------------------

def _augment_one_tile(
    image_path: Path,
    label_path: Path,
    transform: A.Compose,
    n_augmentations: int,
) -> int:
    """Write ``n_augmentations`` photometrically-augmented copies of one tile.

    Args:
        image_path: Path to the source tile image.
        label_path: Path to the matching YOLO label file (copied through
            unchanged for every augmented copy).
        transform: Pre-built Albumentations pipeline.
        n_augmentations: Number of augmented copies to generate.

    Returns:
        Number of augmented tiles successfully written.
    """
    with rasterio.open(image_path) as src:
        image = src.read()  # (C, H, W)
        nodata_value = src.nodata if src.nodata is not None else 0

        # Build an explicit write profile rather than trusting src.profile
        # verbatim - some source TIFFs report blockxsize/blockysize without
        # tiled=True, which GDAL rejects ("BLOCKXSIZE can only be used with
        # TILED=YES").
        block_x = min(256, _round_block_size(src.width))
        block_y = min(256, _round_block_size(src.height))

        profile = {
            "driver": "GTiff",
            "dtype": src.dtypes[0],
            "count": src.count,
            "height": src.height,
            "width": src.width,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "compress": "lzw",
            "predictor": 2,
            "tiled": True,
            "blockxsize": block_x,
            "blockysize": block_y,
        }

    nodata_mask = np.all(image == 0, axis=0)  # (H, W) - fully-black pixels
    image_hwc = np.moveaxis(image, 0, -1).astype(np.float32) / 255.0

    written = 0
    for i in range(n_augmentations):
        augmented_hwc = transform(image=image_hwc)["image"]

        aug_chw = np.moveaxis(augmented_hwc, -1, 0) * 255.0
        aug_chw = np.nan_to_num(aug_chw, nan=0, posinf=255, neginf=0)
        aug_chw[:, nodata_mask] = nodata_value
        aug_chw = np.clip(aug_chw, 0, 255).astype(image.dtype)

        aug_stem = f"{image_path.stem}_aug{i}"
        aug_image_path = image_path.with_name(f"{aug_stem}{image_path.suffix}")
        aug_label_path = label_path.with_name(f"{aug_stem}.txt")

        with rasterio.open(aug_image_path, "w", **profile) as dst:
            dst.write(aug_chw)

        if label_path.exists():
            shutil.copy2(label_path, aug_label_path)
        else:
            aug_label_path.write_text("")

        written += 1

    return written


# ---------------------------------------------------------------------------
# Public entry point - used by scripts/preprocessing.py
# ---------------------------------------------------------------------------

def augment_split(tiled_dir: Path, cfg: Dict) -> int:
    """Generate augmented tile copies for every configured split.

    Args:
        tiled_dir: Root of the tiled dataset (``cfg["paths"]["tiled_dir"]``).
        cfg: Full resolved pipeline config (uses
            ``cfg["data_augmentation"]``).

    Returns:
        Total number of augmented tiles written across all processed splits.
    """
    params = cfg.get("data_augmentation", {})
    if not params.get("enabled", True):
        logger.info("  [augment] disabled - skipping.")
        return 0

    splits = params.get("splits", ["train"])
    n_augmentations = params.get("n_augmentations", 1)
    images_subdir = params.get("images_subdir", cfg["tiling"].get("images_subdir", "images"))
    labels_subdir = params.get("labels_subdir", cfg["tiling"].get("labels_subdir", "labels"))

    transform = _build_transform(params)

    total_written = 0
    for split in splits:
        img_dir = tiled_dir / images_subdir / split
        lbl_dir = tiled_dir / labels_subdir / split

        if not img_dir.exists():
            logger.warning("  [augment] Image directory not found: %s - skipping.", img_dir)
            continue

        image_paths: List[Path] = []
        for pattern in _IMAGE_GLOBS:
            image_paths.extend(img_dir.glob(pattern))
        # Only augment the original tiles, never re-augment an augmented copy.
        image_paths = sorted(p for p in image_paths if "_aug" not in p.stem)

        split_written = 0
        for image_path in image_paths:
            label_path = lbl_dir / f"{image_path.stem}.txt"
            split_written += _augment_one_tile(image_path, label_path, transform, n_augmentations)

        logger.info(
            "  [augment] '%s': %d source tile(s) -> %d augmented tile(s).",
            split, len(image_paths), split_written,
        )
        total_written += split_written

    return total_written