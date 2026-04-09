"""
src/vessels_detect/preprocessing/steps/split.py
------------------------------------------------
Stage 4 - Image-Level Train / Val / Test Split.

Distributes processed GeoTIFF images (and their YOLO label files) into
``train``, ``val``, and ``test`` sub-directories using a **class-aware
greedy assignment** algorithm that prevents spatial leakage and balances
rare-class representation across splits.

Spatial leakage prevention
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each processed image corresponds to a single raw GeoTIFF acquisition.
Because SAHI will slice these images at inference time, mixing spatially
adjacent images across splits is the only leakage risk, and it is fully
prevented by assigning at the **image level** (one image → one split).

No background dropping
~~~~~~~~~~~~~~~~~~~~~~~
Background images are kept in all splits.  YOLO's ``mosaic`` and
``mixup`` augmentations provide implicit hard-negative mining at training
time; removing background images here would reduce their effectiveness.

Class-aware greedy assignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.  Scan all label ``.txt`` files and build a per-image annotation profile
    (class → count).
2.  Sort images by total priority-class count (descending) so the most
    class-constraining images are assigned first.
3.  Score each eligible split by how much adding this image would reduce
    the per-class deficit weighted by ``priority_weight``.
4.  Assign to the best-scoring split; enforce integer capacity caps derived
    from the requested ratios to prevent collapse into train.

Output layout
~~~~~~~~~~~~~
::

    <dataset_dir>/
        images/
            train/  *.tif
            val/    *.tif
            test/   *.tif
        labels/
            train/  *.txt
            val/    *.txt
            test/   *.txt

Typical usage (via manager)::

    from src.vessels_detect.preprocessing.steps.split import SplitStep
    step = SplitStep()
    step.run(cfg)
"""

from __future__ import annotations

import logging
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.vessels_detect.preprocessing.steps.base import BaseStep

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Hyperparameters for :class:`SplitStep`.

    Attributes:
        train_ratio: Target fraction of images for the training set.
        val_ratio: Target fraction for validation.
        test_ratio: Target fraction for testing.
            Must sum to 1.0.
        priority_class_ids: Class IDs given extra weight in the deficit
            scoring function to preserve rare-class representation.
        priority_weight: Multiplicative weight for priority classes.
            Values in [2, 8] work well; higher → more aggressive balancing.
        random_seed: Seed for reproducible tie-breaking.
        copy: If ``True``, files are copied to the dataset directory and
            originals are preserved.  If ``False``, files are moved.
    """

    train_ratio:        float     = 0.70
    val_ratio:          float     = 0.15
    test_ratio:         float     = 0.15
    priority_class_ids: List[int] = field(default_factory=lambda: [0, 1])
    priority_weight:    float     = 5.0
    random_seed:        int       = 42
    copy:               bool      = False

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.6f}."
            )
        if self.priority_weight < 1.0:
            raise ValueError(
                f"priority_weight must be ≥ 1.0, got {self.priority_weight}."
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "SplitConfig":
        """Construct from a YAML section dictionary.

        Args:
            cfg: Dictionary with keys matching the dataclass field names.

        Returns:
            A populated :class:`SplitConfig` instance.
        """
        return cls(
            train_ratio=cfg.get("train_ratio", 0.70),
            val_ratio=cfg.get("val_ratio", 0.15),
            test_ratio=cfg.get("test_ratio", 0.15),
            priority_class_ids=cfg.get("priority_class_ids", [0, 1]),
            priority_weight=cfg.get("priority_weight", 5.0),
            random_seed=cfg.get("random_seed", 42),
            copy=cfg.get("copy", False),
        )


# ---------------------------------------------------------------------------
# Annotation profile helpers
# ---------------------------------------------------------------------------

def _read_label_profile(label_path: Path) -> Dict[int, int]:
    """Count per-class annotation instances in one YOLO label file.

    Args:
        label_path: Path to a YOLO OBB ``.txt`` label file.

    Returns:
        Dictionary mapping ``class_id`` (int) to instance count.
        Empty dict for background (empty) label files.
    """
    profile: Dict[int, int] = defaultdict(int)
    if not label_path.exists():
        return profile

    try:
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if line:
                cls_id = int(line.split()[0])
                profile[cls_id] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cannot parse label '%s': %s", label_path.name, exc)

    return dict(profile)


def _build_image_profiles(
    tif_files: List[Path],
    labels_dir: Path,
) -> Dict[str, Dict[int, int]]:
    """Build annotation profiles for all images.

    Args:
        tif_files: List of processed GeoTIFF paths.
        labels_dir: Directory containing corresponding ``.txt`` label files.

    Returns:
        Mapping ``{image_stem: {class_id: count}}``.
    """
    profiles: Dict[str, Dict[int, int]] = {}
    for tif in tif_files:
        label_path  = labels_dir / f"{tif.stem}.txt"
        profiles[tif.stem] = _read_label_profile(label_path)
    return profiles


# ---------------------------------------------------------------------------
# Capacity and scoring
# ---------------------------------------------------------------------------

def _compute_capacities(
    n_images: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    """Compute integer image-count capacity caps per split.

    Uses floor allocation with remainder distributed to larger ratios.

    Args:
        n_images: Total number of images to distribute.
        train_ratio: Target train fraction.
        val_ratio: Target val fraction.
        test_ratio: Target test fraction.

    Returns:
        Dictionary ``{"train": n, "val": n, "test": n}`` where values sum
        to *n_images*.
    """
    raw = {
        "train": train_ratio * n_images,
        "val":   val_ratio   * n_images,
        "test":  test_ratio  * n_images,
    }
    caps = {k: math.floor(v) for k, v in raw.items()}
    remainder = n_images - sum(caps.values())

    # Distribute remainder to splits with the largest fractional parts.
    fracs = sorted(raw, key=lambda k: raw[k] - caps[k], reverse=True)
    for k in fracs[:remainder]:
        caps[k] += 1

    return caps


def _score_assignment(
    profile:            Dict[int, int],
    current_counts:     Dict[str, Dict[int, int]],
    targets:            Dict[str, Dict[int, int]],
    priority_class_ids: List[int],
    priority_weight:    float,
    split:              str,
) -> float:
    """Score how beneficial it would be to assign *image* to *split*.

    A higher score means the assignment reduces the class deficit more.

    Args:
        profile: Per-class annotation counts for the candidate image.
        current_counts: Running per-split per-class counts.
        targets: Target per-split per-class counts (float as dict).
        priority_class_ids: Class IDs that receive extra weight.
        priority_weight: Multiplicative weight for priority classes.
        split: Name of the split being evaluated.

    Returns:
        Scalar score (higher is better).
    """
    score = 0.0
    for cls_id, count in profile.items():
        deficit = targets[split].get(cls_id, 0.0) - current_counts[split].get(cls_id, 0)
        w = priority_weight if cls_id in priority_class_ids else 1.0
        score += w * min(deficit, count)
    return score


# ---------------------------------------------------------------------------
# Greedy assignment
# ---------------------------------------------------------------------------

def _assign_splits(
    stems:              List[str],
    profiles:           Dict[str, Dict[int, int]],
    caps:               Dict[str, int],
    train_ratio:        float,
    val_ratio:          float,
    test_ratio:         float,
    priority_class_ids: List[int],
    priority_weight:    float,
    rng:                random.Random,
) -> Dict[str, str]:
    """Class-aware greedy split assignment.

    Args:
        stems: Image stem names to assign.
        profiles: Per-image annotation profiles.
        caps: Integer image-count capacity caps per split.
        train_ratio: Train fraction (used for target counts).
        val_ratio: Val fraction.
        test_ratio: Test fraction.
        priority_class_ids: Class IDs with extra scoring weight.
        priority_weight: Multiplier for priority classes.
        rng: Seeded random instance for tie-breaking.

    Returns:
        Mapping ``{image_stem: split_name}``.
    """
    # Compute aggregate annotation targets per split per class.
    total_class_counts: Dict[int, int] = defaultdict(int)
    for profile in profiles.values():
        for cls_id, cnt in profile.items():
            total_class_counts[cls_id] += cnt

    ratio_map = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    targets: Dict[str, Dict[int, float]] = {
        split: {cls_id: cnt * ratio_map[split] for cls_id, cnt in total_class_counts.items()}
        for split in _SPLITS
    }

    current_counts: Dict[str, Dict[int, int]] = {s: defaultdict(int) for s in _SPLITS}
    assignment_counts: Dict[str, int] = {s: 0 for s in _SPLITS}
    assignment: Dict[str, str] = {}

    # Process images with most priority-class annotations first.
    def _priority_key(stem: str) -> int:
        p = profiles.get(stem, {})
        return sum(p.get(c, 0) for c in priority_class_ids)

    ordered = sorted(stems, key=_priority_key, reverse=True)

    for stem in ordered:
        profile  = profiles.get(stem, {})
        eligible = [s for s in _SPLITS if assignment_counts[s] < caps[s]]

        if not eligible:
            # Fallback: assign to split with most remaining capacity.
            eligible = sorted(_SPLITS, key=lambda s: caps[s] - assignment_counts[s], reverse=True)

        if not eligible:
            eligible = ["train"]

        scores = {
            s: _score_assignment(
                profile, current_counts, targets,
                priority_class_ids, priority_weight, s
            )
            for s in eligible
        }

        best_score = max(scores.values())
        best_split = rng.choice(
            [s for s, sc in scores.items() if sc == best_score]
        )

        assignment[stem] = best_split
        assignment_counts[best_split] += 1
        for cls_id, cnt in profile.items():
            current_counts[best_split][cls_id] += cnt

    return assignment


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _transfer_file(src: Path, dst: Path, copy: bool) -> None:
    """Copy or move *src* to *dst*, creating parent directories as needed.

    Args:
        src: Source file path.
        dst: Destination file path.
        copy: If ``True``, copy; otherwise move.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        logger.debug("Source does not exist, skipping: %s", src)
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


# ---------------------------------------------------------------------------
# Public step
# ---------------------------------------------------------------------------

class SplitStep(BaseStep):
    """Distribute processed images and labels into train / val / test splits.

    Reads from:
        * ``cfg["paths"]["spatial_dir"]`` - processed GeoTIFFs.
        * ``cfg["paths"]["labels_dir"]`` - global YOLO label ``.txt`` files.

    Writes to:
        * ``cfg["paths"]["dataset_dir"]/images/{train,val,test}/``
        * ``cfg["paths"]["dataset_dir"]/labels/{train,val,test}/``

    Files are moved by default (``copy: false`` in config) to save disk
    space.  Set ``copy: true`` to preserve the originals.
    """

    NAME = "split"

    def run(self, cfg: dict) -> None:
        """Execute the image-level split stage.

        Args:
            cfg: Resolved configuration dictionary.  Uses:

                * ``cfg["paths"]["spatial_dir"]``
                * ``cfg["paths"]["labels_dir"]``
                * ``cfg["paths"]["dataset_dir"]``
                * ``cfg["split"]`` - stage hyperparameters.
        """
        paths  = cfg["paths"]
        params = SplitConfig.from_dict(cfg.get("split", {}))

        img_dir:     Path = paths["spatial_dir"]
        labels_dir:  Path = paths["labels_dir"]
        dataset_dir: Path = paths["dataset_dir"]

        tif_files = sorted(img_dir.glob("*.tif"))
        if not tif_files:
            raise RuntimeError(f"No .tif files found in '{img_dir}'.")

        n = len(tif_files)
        logger.info(
            "Splitting %d image(s): train=%.0f%%  val=%.0f%%  test=%.0f%%",
            n,
            params.train_ratio * 100,
            params.val_ratio   * 100,
            params.test_ratio  * 100,
        )

        # ── Build annotation profiles ──────────────────────────────────────
        profiles = _build_image_profiles(tif_files, labels_dir)
        stems    = [tif.stem for tif in tif_files]

        n_annotated = sum(1 for p in profiles.values() if p)
        logger.info(
            "  %d / %d image(s) carry at least one annotation.",
            n_annotated, n,
        )

        # ── Compute capacities and assign ──────────────────────────────────
        caps = _compute_capacities(
            n, params.train_ratio, params.val_ratio, params.test_ratio
        )
        logger.info("  Capacity caps: %s", caps)

        rng = random.Random(params.random_seed)
        assignment = _assign_splits(
            stems, profiles, caps,
            params.train_ratio, params.val_ratio, params.test_ratio,
            params.priority_class_ids, params.priority_weight, rng,
        )

        # ── Log distribution ───────────────────────────────────────────────
        split_counts: Dict[str, int] = defaultdict(int)
        for split in assignment.values():
            split_counts[split] += 1

        for split in _SPLITS:
            logger.info("  %-5s: %d image(s)", split, split_counts.get(split, 0))

        # ── Transfer files ─────────────────────────────────────────────────
        action = "Copying" if params.copy else "Moving"
        logger.info("%s files to '%s' …", action, dataset_dir)

        for tif in tif_files:
            split      = assignment[tif.stem]
            dst_img    = dataset_dir / "images" / split / tif.name
            dst_lbl    = dataset_dir / "labels" / split / f"{tif.stem}.txt"
            src_lbl    = labels_dir / f"{tif.stem}.txt"

            _transfer_file(tif,      dst_img, params.copy)
            _transfer_file(src_lbl,  dst_lbl, params.copy)

        logger.info(
            "Split stage complete.  Output layout: '%s'.", dataset_dir
        )
        self._log_class_summary(assignment, profiles)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_class_summary(
        assignment: Dict[str, str],
        profiles:   Dict[str, Dict[int, int]],
    ) -> None:
        """Log per-split per-class annotation counts for diagnostics.

        Args:
            assignment: Mapping ``{image_stem: split_name}``.
            profiles: Per-image annotation profiles.
        """
        summary: Dict[str, Dict[int, int]] = {s: defaultdict(int) for s in _SPLITS}
        for stem, split in assignment.items():
            for cls_id, cnt in profiles.get(stem, {}).items():
                summary[split][cls_id] += cnt

        all_classes = sorted(
            {cls_id for sp in summary.values() for cls_id in sp}
        )
        if not all_classes:
            logger.info("  No annotations found across any split.")
            return

        logger.info("  Per-split class distribution:")
        header = f"  {'class':>6}" + "".join(f"  {s:>8}" for s in _SPLITS)
        logger.info(header)
        for cls_id in all_classes:
            row = f"  {cls_id:>6}" + "".join(
                f"  {summary[s].get(cls_id, 0):>8}" for s in _SPLITS
            )
            logger.info(row)
