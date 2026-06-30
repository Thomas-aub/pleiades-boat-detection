"""
src/vessels_detect/preprocessing/background_filtering.py
------------------------------------------------------------
Step 5 - Background tile filtering.

Caps the fraction of background (empty-label) tiles in a tiled split by
relocating the excess into a ``moved/`` sub-directory. Files are moved, not
deleted, so the operation is fully reversible.

This step only reads label files (small text) to decide which tiles are
background; image tiles themselves are never opened, so memory use stays
flat no matter how large the tiled dataset is.
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_IMAGE_GLOBS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_background(label_path: Path) -> bool:
    """A tile is background if its label file is missing or has no annotation lines."""
    if not label_path.exists():
        return True
    return label_path.read_text(encoding="utf-8").strip() == ""


def _collect_tiles(img_dir: Path, lbl_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Partition tiles in *img_dir* into (positives, backgrounds)."""
    image_paths: List[Path] = []
    for pattern in _IMAGE_GLOBS:
        image_paths.extend(img_dir.glob(pattern))

    positives, backgrounds = [], []
    for img_path in image_paths:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        (backgrounds if _is_background(label_path) else positives).append((img_path, label_path))
    return positives, backgrounds


def _compute_bg_keep(num_positives: int, target_ratio: float) -> int:
    """bg_keep = floor(P * r / (1 - r)), derived from ratio = bg_keep / (P + bg_keep)."""
    return int(num_positives * (target_ratio / (1.0 - target_ratio)))


def _move_pair(img_path: Path, lbl_path: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
    shutil.move(str(img_path), str(dst_img_dir / img_path.name))
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(dst_lbl_dir / lbl_path.name))


# ---------------------------------------------------------------------------
# Public entry point - used by scripts/preprocessing.py
# ---------------------------------------------------------------------------

def filter_background(tiled_dir: Path, cfg: Dict) -> int:
    """Cap the background-tile ratio for every configured split.

    Args:
        tiled_dir: Root of the tiled dataset (``cfg["paths"]["tiled_dir"]``).
        cfg: Full resolved pipeline config (uses
            ``cfg["background_reduction"]``).

    Returns:
        Total number of background tiles moved across all processed splits.
    """
    params = cfg["background_reduction"]
    splits = params.get("splits", ["train"])
    target_bg_ratio = params.get("target_bg_ratio", 0.15)
    random_seed = params.get("random_seed", 42)
    images_subdir = params.get("images_subdir", "images")
    labels_subdir = params.get("labels_subdir", "labels")
    moved_subdir = params.get("moved_subdir", "moved")

    total_moved = 0

    for split in splits:
        img_dir = tiled_dir / images_subdir / split
        lbl_dir = tiled_dir / labels_subdir / split

        if not img_dir.exists():
            logger.warning("  [bg-filter] Image directory not found: %s - skipping.", img_dir)
            continue

        positives, backgrounds = _collect_tiles(img_dir, lbl_dir)
        num_pos, num_bg = len(positives), len(backgrounds)

        if num_pos == 0:
            logger.warning("  [bg-filter] No positive tiles in '%s' - skipping.", split)
            continue

        bg_keep = _compute_bg_keep(num_pos, target_bg_ratio)
        num_to_move = max(0, num_bg - bg_keep)

        if num_to_move == 0:
            logger.info("  [bg-filter] '%s' already within target ratio.", split)
            continue

        random.seed(random_seed)
        random.shuffle(backgrounds)
        to_move = backgrounds[bg_keep:]

        dst_img_dir = tiled_dir / moved_subdir / images_subdir / split
        dst_lbl_dir = tiled_dir / moved_subdir / labels_subdir / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in to_move:
            _move_pair(img_path, lbl_path, dst_img_dir, dst_lbl_dir)

        logger.info(
            "  [bg-filter] '%s': moved %d background tile(s), kept %d (target %.0f%%).",
            split, len(to_move), bg_keep, target_bg_ratio * 100,
        )
        total_moved += len(to_move)

    return total_moved
