"""
src/vessels_detect/preprocessing/dataset_split.py
----------------------------------------------------
Step 3 - Image-level train / val / test split.

Distributes enhanced images (and their YOLO label files) into ``train``,
``val``, ``test`` sub-directories using a class-aware greedy assignment that
balances rare-class representation while keeping every image whole (so a
single acquisition never leaks across splits).

This step only ever touches small label-profile dictionaries and file
paths - no raster pixel data is read at all - so memory use is independent
of image size or dataset size.
"""

from __future__ import annotations

import logging
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Annotation profile helpers
# ---------------------------------------------------------------------------

def _read_label_profile(label_path: Path) -> Dict[int, int]:
    """Count per-class annotation instances in one YOLO label file."""
    profile: Dict[int, int] = defaultdict(int)
    if not label_path.exists():
        return profile
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if line:
            profile[int(line.split()[0])] += 1
    return dict(profile)


def _compute_capacities(n_images: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    """Integer per-split capacity caps that sum exactly to *n_images*."""
    raw = {"train": train_ratio * n_images, "val": val_ratio * n_images, "test": test_ratio * n_images}
    caps = {k: math.floor(v) for k, v in raw.items()}
    remainder = n_images - sum(caps.values())
    for k in sorted(raw, key=lambda k: raw[k] - caps[k], reverse=True)[:remainder]:
        caps[k] += 1
    return caps


def _score_assignment(
    profile: Dict[int, int],
    current_counts: Dict[str, Dict[int, int]],
    targets: Dict[str, Dict[int, float]],
    priority_class_ids: List[int],
    priority_weight: float,
    split: str,
) -> float:
    """Higher score = assigning this image to *split* reduces the class deficit more."""
    score = 0.0
    for cls_id, count in profile.items():
        deficit = targets[split].get(cls_id, 0.0) - current_counts[split].get(cls_id, 0)
        weight = priority_weight if cls_id in priority_class_ids else 1.0
        score += weight * min(deficit, count)
    return score


def _assign_splits(
    stems: List[str],
    profiles: Dict[str, Dict[int, int]],
    caps: Dict[str, int],
    ratios: Dict[str, float],
    priority_class_ids: List[int],
    priority_weight: float,
    rng: random.Random,
) -> Dict[str, str]:
    """Class-aware greedy split assignment. Returns ``{image_stem: split_name}``."""
    total_class_counts: Dict[int, int] = defaultdict(int)
    for profile in profiles.values():
        for cls_id, cnt in profile.items():
            total_class_counts[cls_id] += cnt

    targets: Dict[str, Dict[int, float]] = {
        split: {cls_id: cnt * ratios[split] for cls_id, cnt in total_class_counts.items()}
        for split in _SPLITS
    }
    current_counts: Dict[str, Dict[int, int]] = {s: defaultdict(int) for s in _SPLITS}
    assignment_counts: Dict[str, int] = {s: 0 for s in _SPLITS}
    assignment: Dict[str, str] = {}

    def _priority_key(stem: str) -> int:
        p = profiles.get(stem, {})
        return sum(p.get(c, 0) for c in priority_class_ids)

    for stem in sorted(stems, key=_priority_key, reverse=True):
        profile = profiles.get(stem, {})
        eligible = [s for s in _SPLITS if assignment_counts[s] < caps[s]] or list(_SPLITS)

        scores = {
            s: _score_assignment(profile, current_counts, targets, priority_class_ids, priority_weight, s)
            for s in eligible
        }
        best_score = max(scores.values())
        best_split = rng.choice([s for s, sc in scores.items() if sc == best_score])

        assignment[stem] = best_split
        assignment_counts[best_split] += 1
        for cls_id, cnt in profile.items():
            current_counts[best_split][cls_id] += cnt

    return assignment


def _transfer_file(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


# ---------------------------------------------------------------------------
# Public entry point - used by scripts/preprocessing.py
# ---------------------------------------------------------------------------

def split_dataset(
    image_label_pairs: List[Tuple[Path, Path]],
    out_dir: Path,
    cfg: Dict,
) -> Dict[str, str]:
    """Assign every (image, label) pair to train/val/test and move/copy the files.

    Args:
        image_label_pairs: List of ``(enhanced_image_path, label_path)``
            pairs, one per source acquisition.
        out_dir: Root output directory; files land in
            ``out_dir/images/{split}/`` and ``out_dir/labels/{split}/``.
        cfg: Full resolved pipeline config (uses ``cfg["split"]``).

    Returns:
        Mapping ``{image_stem: split_name}``.
    """
    params = cfg["split"]
    train_ratio = params.get("train_ratio", 0.70)
    val_ratio = params.get("val_ratio", 0.15)
    test_ratio = params.get("test_ratio", 0.15)
    priority_class_ids = params.get("priority_class_ids", [])
    priority_weight = params.get("priority_weight", 5.0)
    random_seed = params.get("random_seed", 42)
    copy = params.get("copy", False)

    profiles = {img.stem: _read_label_profile(lbl) for img, lbl in image_label_pairs}
    stems = [img.stem for img, _ in image_label_pairs]

    caps = _compute_capacities(len(stems), train_ratio, val_ratio, test_ratio)
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    rng = random.Random(random_seed)
    assignment = _assign_splits(stems, profiles, caps, ratios, priority_class_ids, priority_weight, rng)

    split_counts: Dict[str, int] = defaultdict(int)
    for img, lbl in image_label_pairs:
        split = assignment[img.stem]
        split_counts[split] += 1
        _transfer_file(img, out_dir / "images" / split / img.name, copy)
        _transfer_file(lbl, out_dir / "labels" / split / f"{img.stem}.txt", copy)

    logger.info("  [split] Distribution: %s", dict(split_counts))
    return assignment
