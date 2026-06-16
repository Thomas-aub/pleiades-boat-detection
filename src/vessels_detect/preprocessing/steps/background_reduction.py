"""
src/vessels_detect/preprocessing/steps/background_reduction.py
---------------------------------------------------------------
Stage 6 - Background tile reduction.

Curates the tiled dataset produced by Stage 5 by relocating excess
background (empty) tiles into a ``moved/`` subdirectory, ensuring that
background images make up at most a configurable fraction of each processed
split.

What counts as a "background" tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A tile is considered background when its matching YOLO label file either:

- does not exist at all, or
- exists but contains only whitespace (zero annotation lines).

Reduction strategy
~~~~~~~~~~~~~~~~~~
Given *P* positive (non-background) tiles and a target ratio *r*:

.. math::

    \\text{bg\\_keep} = \\lfloor P \\cdot r / (1 - r) \\rfloor

Excess tiles beyond ``bg_keep`` are moved (not deleted) so the operation is
fully reversible.  Tiles are shuffled with a fixed seed before selection so
the kept subset is spatially distributed rather than biased toward one corner.

Configuration path in the YAML (``cfg["background_reduction"]``)::

    background_reduction:
      splits:          [train, val]   # which splits to process
      target_bg_ratio: 0.15           # max background fraction (0–1, exclusive)
      random_seed:     42             # reproducibility seed for tile shuffle
      images_subdir:   images         # subfolder name inside tiled_dir
      labels_subdir:   labels         # subfolder name inside tiled_dir
      moved_subdir:    moved          # where excess tiles are relocated

Input layout (Stage 6 / split output)::

    tiled_dir/
      {images_subdir}/{fold_name}/   *.tif   (e.g. fold_00/, fold_01/, ...)
      {labels_subdir}/{fold_name}/   *.txt

Output layout (files physically moved, not copied)::

    tiled_dir/
      {images_subdir}/{fold_name}/   <retained tiles>
      {labels_subdir}/{fold_name}/   <retained labels>
      {moved_subdir}/
        {images_subdir}/{fold_name}/   <excess background tiles>
        {labels_subdir}/{fold_name}/   <excess background labels>

Typical standalone usage::

    from pathlib import Path
    from src.vessels_detect.preprocessing.steps.background_reduction import BackgroundReductionStep

    step = BackgroundReductionStep()
    step.run(cfg)   # cfg is the fully-resolved config dict from manager.py
"""

from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from src.vessels_detect.preprocessing.steps.base import BaseStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BackgroundReductionConfig:
    """All hyperparameters consumed by :class:`BackgroundReductionStep`.

    Attributes:
        splits: Dataset splits to process.  Each name must correspond to a
            sub-directory inside ``tiled_dir/{images_subdir}/``.
        target_bg_ratio: Maximum fraction of background (empty) tiles
            allowed in the final dataset.  Must be in the open interval
            ``(0.0, 1.0)``.  Example: ``0.15`` keeps at most 15 % background.
        random_seed: Seed passed to :func:`random.seed` before shuffling
            background tiles, ensuring the kept subset is reproducible and
            spatially distributed.
        images_subdir: Subfolder name used for tile images inside
            ``tiled_dir``.  Must match the value used in Stage 5.
        labels_subdir: Subfolder name used for tile label files inside
            ``tiled_dir``.  Must match the value used in Stage 5.
        moved_subdir: Name of the top-level subdirectory inside ``tiled_dir``
            where excess background tiles are relocated.
    """

    splits: List[str] = field(default_factory=lambda: ["fold_00"])
    target_bg_ratio: float = 0.15
    random_seed: int = 42
    images_subdir: str = "images"
    labels_subdir: str = "labels"
    moved_subdir: str = "moved"

    def __post_init__(self) -> None:
        if not (0.0 < self.target_bg_ratio < 1.0):
            raise ValueError(
                f"target_bg_ratio must be in the open interval (0.0, 1.0), "
                f"got {self.target_bg_ratio}."
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "BackgroundReductionConfig":
        """Construct from the ``cfg["background_reduction"]`` sub-dictionary.

        Args:
            cfg: The ``background_reduction`` section parsed from the YAML
                config.  Unknown keys are silently ignored.

        Returns:
            A fully populated :class:`BackgroundReductionConfig` instance.
        """
        return cls(
            splits=list(cfg.get("splits", ["fold_00"])),
            target_bg_ratio=float(cfg.get("target_bg_ratio", 0.15)),
            random_seed=int(cfg.get("random_seed", 42)),
            images_subdir=str(cfg.get("images_subdir", "images")),
            labels_subdir=str(cfg.get("labels_subdir", "labels")),
            moved_subdir=str(cfg.get("moved_subdir", "moved")),
        )


# ---------------------------------------------------------------------------
# Pure helpers (no side-effects, easy to unit-test)
# ---------------------------------------------------------------------------

_IMAGE_GLOBS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def _is_background(label_path: Path) -> bool:
    """Return ``True`` when *label_path* represents a background (empty) tile.

    Args:
        label_path: Expected path of the YOLO label file (may not exist).

    Returns:
        ``True`` if the label file is absent or contains no annotation lines.
    """
    if not label_path.exists():
        return True
    return label_path.read_text(encoding="utf-8").strip() == ""


def _collect_tiles(
    img_dir: Path,
    lbl_dir: Path,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Partition tiles in *img_dir* into positives and backgrounds.

    Args:
        img_dir: Directory containing tile images.
        lbl_dir: Directory containing matching YOLO label files.

    Returns:
        A pair ``(positives, backgrounds)`` where each element is a list of
        ``(image_path, label_path)`` tuples.  The label path is always
        ``lbl_dir / stem + ".txt"`` regardless of whether the file exists.
    """
    image_paths: List[Path] = []
    for pattern in _IMAGE_GLOBS:
        image_paths.extend(img_dir.glob(pattern))

    positives: List[Tuple[Path, Path]] = []
    backgrounds: List[Tuple[Path, Path]] = []

    for img_path in image_paths:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if _is_background(label_path):
            backgrounds.append((img_path, label_path))
        else:
            positives.append((img_path, label_path))

    return positives, backgrounds


def _compute_bg_keep(num_positives: int, target_ratio: float) -> int:
    """Compute how many background tiles to retain.

    Derived from ``ratio = bg_keep / (positives + bg_keep)``:

    .. math::

        \\text{bg\\_keep} = \\lfloor P \\cdot r / (1 - r) \\rfloor

    Args:
        num_positives: Number of tiles containing at least one annotation.
        target_ratio: Target background fraction in ``(0, 1)``.

    Returns:
        Maximum number of background tiles to keep (≥ 0).
    """
    return int(num_positives * (target_ratio / (1.0 - target_ratio)))


def _move_pair(img_path: Path, lbl_path: Path, dst_img_dir: Path, dst_lbl_dir: Path) -> None:
    """Relocate one image/label pair to the ``moved`` destination directories.

    The label file is moved only when it physically exists; a missing label
    file (implicit background) is silently skipped.

    Args:
        img_path: Source image path.
        lbl_path: Expected source label path (may not exist).
        dst_img_dir: Destination directory for the image.
        dst_lbl_dir: Destination directory for the label.
    """
    shutil.move(str(img_path), str(dst_img_dir / img_path.name))
    if lbl_path.exists():
        shutil.move(str(lbl_path), str(dst_lbl_dir / lbl_path.name))


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

class BackgroundReductionStep(BaseStep):
    """Stage 6 - background tile reduction.

    Reads the tiled dataset produced by :class:`TilingStep` and relocates
    excess background (empty-label) tiles to a ``moved/`` subdirectory,
    capping the background fraction at ``target_bg_ratio``.

    Only folds listed in ``cfg["background_reduction"]["splits"]`` are
    touched.  Folds not listed are left intact, preserving their true class
    distribution (useful for held-out evaluation folds).

    Files are *moved*, not deleted, so the operation is fully reversible:
    restore the originals by moving them back from the ``moved/`` tree.

    Input layout (Stage 6 / split output)::

        tiled_dir/
          {images_subdir}/{fold_name}/   *.tif   (e.g. fold_00/, fold_01/, ...)
          {labels_subdir}/{fold_name}/   *.txt

    Output layout::

        tiled_dir/
          {images_subdir}/{fold_name}/   <retained tiles>
          {labels_subdir}/{fold_name}/   <retained labels>
          {moved_subdir}/
            {images_subdir}/{fold_name}/   <excess background tiles>
            {labels_subdir}/{fold_name}/   <excess background labels>
    """

    NAME = "background_reduction"

    # ------------------------------------------------------------------
    # Restore helpers (inverse of run — used by the CV experiment runner)
    # ------------------------------------------------------------------

    @staticmethod
    def restore_split(
        split: str,
        tiled_dir: Path,
        step_cfg: "BackgroundReductionConfig",
    ) -> int:
        """Move all excess-background tiles back from ``moved/`` to their fold.

        This is the exact inverse of :meth:`_process_split`.  Call it before
        re-applying background reduction for a new CV iteration so that each
        iteration starts from the full, unfiltered tile pool.

        Args:
            split:     Fold name (e.g. ``"fold_00"``).
            tiled_dir: Root of the tiled dataset.
            step_cfg:  Resolved step configuration.

        Returns:
            Number of tiles restored for this split (image files only).
        """
        src_img_dir = tiled_dir / step_cfg.moved_subdir / step_cfg.images_subdir / split
        src_lbl_dir = tiled_dir / step_cfg.moved_subdir / step_cfg.labels_subdir / split
        dst_img_dir = tiled_dir / step_cfg.images_subdir / split
        dst_lbl_dir = tiled_dir / step_cfg.labels_subdir / split

        if not src_img_dir.exists():
            logger.debug("  No moved tiles for split %s — nothing to restore.", split)
            return 0

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in src_img_dir.iterdir():
            if img_path.is_file():
                shutil.move(str(img_path), str(dst_img_dir / img_path.name))
                count += 1

        if src_lbl_dir.exists():
            for lbl_path in src_lbl_dir.iterdir():
                if lbl_path.is_file():
                    shutil.move(str(lbl_path), str(dst_lbl_dir / lbl_path.name))

        logger.info(
            "  [%s] Restored %d tile(s) from moved/ → %s",
            split, count, dst_img_dir,
        )
        return count

    @classmethod
    def restore(
        cls,
        splits: List[str],
        tiled_dir: Path,
        step_cfg: "BackgroundReductionConfig",
    ) -> int:
        """Restore all ``moved/`` backgrounds for every split in *splits*.

        Intended to be called by the CV experiment runner at the start of each
        new outer iteration, before re-applying background reduction with the
        new val folds excluded.

        Args:
            splits:    Fold names to restore (typically all folds).
            tiled_dir: Root of the tiled dataset.
            step_cfg:  Resolved step configuration (needs ``moved_subdir``,
                       ``images_subdir``, ``labels_subdir``).

        Returns:
            Total number of image tiles restored across all splits.
        """
        logger.info("Restoring moved backgrounds for splits: %s", splits)
        total = 0
        for split in splits:
            total += cls.restore_split(split, tiled_dir, step_cfg)
        logger.info("Restore complete. Total tiles restored: %d", total)
        return total

    # ------------------------------------------------------------------
    # BaseStep interface
    # ------------------------------------------------------------------

    def run(self, cfg: dict) -> None:
        """Execute the background-reduction step.

        Args:
            cfg: Fully resolved configuration dictionary from
                :func:`~manager.load_config`.  Expected keys:

                ``cfg["paths"]["tiled_dir"]``
                    Root of the tiled dataset (output of Stage 6 / split).

                ``cfg["background_reduction"]``
                    Step-specific hyperparameters; see
                    :class:`BackgroundReductionConfig`.

        Raises:
            KeyError: If required config keys are absent.
            ValueError: If :class:`BackgroundReductionConfig` validation fails.
        """
        step_cfg = BackgroundReductionConfig.from_dict(
            cfg.get("background_reduction", {})
        )
        tiled_dir: Path = cfg["paths"]["tiled_dir"]

        logger.info("Background reduction step configuration:")
        logger.info("  tiled_dir        : %s", tiled_dir)
        logger.info("  splits           : %s", step_cfg.splits)
        logger.info("  target_bg_ratio  : %.2f (%.1f %%)", step_cfg.target_bg_ratio, step_cfg.target_bg_ratio * 100)
        logger.info("  random_seed      : %d", step_cfg.random_seed)
        logger.info("  images_subdir    : %s", step_cfg.images_subdir)
        logger.info("  labels_subdir    : %s", step_cfg.labels_subdir)
        logger.info("  moved_subdir     : %s", step_cfg.moved_subdir)

        total_moved = 0

        for split in step_cfg.splits:
            moved = self._process_split(split, tiled_dir, step_cfg)
            total_moved += moved

        logger.info("")
        logger.info(
            "Background reduction complete.  Total tiles moved: %d.  "
            "Delete any .cache files before restarting YOLO training.",
            total_moved,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process_split(
        split: str,
        tiled_dir: Path,
        step_cfg: BackgroundReductionConfig,
    ) -> int:
        """Process a single dataset split.

        Args:
            split: Fold name (e.g. ``"fold_00"``).
            tiled_dir: Root of the tiled dataset.
            step_cfg: Resolved step configuration.

        Returns:
            Number of background tiles moved for this split.
        """
        logger.info("")
        logger.info("[%s SPLIT]", split.upper())

        img_dir = tiled_dir / step_cfg.images_subdir / split
        lbl_dir = tiled_dir / step_cfg.labels_subdir / split

        if not img_dir.exists():
            logger.warning("  Image directory not found: %s — skipping.", img_dir)
            return 0

        positives, backgrounds = _collect_tiles(img_dir, lbl_dir)

        num_pos = len(positives)
        num_bg  = len(backgrounds)
        total   = num_pos + num_bg

        logger.info(
            "  Found %d tile(s): %d with annotations, %d background.",
            total, num_pos, num_bg,
        )

        if num_pos == 0:
            logger.warning(
                "  No positive tiles found — skipping to avoid removing all data."
            )
            return 0

        bg_keep     = _compute_bg_keep(num_pos, step_cfg.target_bg_ratio)
        num_to_move = max(0, num_bg - bg_keep)

        logger.info(
            "  Target ratio %.1f %%: keep %d background tile(s), move %d.",
            step_cfg.target_bg_ratio * 100, bg_keep, num_to_move,
        )

        if num_to_move == 0:
            logger.info(
                "  Already at or below target ratio (actual %.1f %%) — nothing to move.",
                100.0 * num_bg / total if total else 0.0,
            )
            return 0

        # Shuffle for spatial distribution, then slice the excess.
        random.seed(step_cfg.random_seed)
        random.shuffle(backgrounds)
        to_move = backgrounds[bg_keep:]

        # Ensure destination directories exist.
        dst_img_dir = tiled_dir / step_cfg.moved_subdir / step_cfg.images_subdir / split
        dst_lbl_dir = tiled_dir / step_cfg.moved_subdir / step_cfg.labels_subdir / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in to_move:
            _move_pair(img_path, lbl_path, dst_img_dir, dst_lbl_dir)

        logger.info(
            "  Moved %d background tile(s) → %s",
            len(to_move), dst_img_dir,
        )
        logger.info(
            "  New split size: %d tile(s) (%d annotations, %d background = %.1f %%).",
            num_pos + bg_keep,
            num_pos,
            bg_keep,
            100.0 * bg_keep / (num_pos + bg_keep) if (num_pos + bg_keep) else 0.0,
        )

        return len(to_move)