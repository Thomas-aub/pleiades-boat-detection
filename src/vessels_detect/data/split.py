"""
src/vessels_detect/data/split.py
---------------------------------
Splits tiles into train / val / test partitions with zero spatial leakage,
then optionally balances background (empty) tiles within each partition.

**Spatial Leakage Prevention**
    Tiles derived from the same source GeoTIFF spatially overlap and share
    geographic context.  Mixing them across partitions would allow the model
    to see validation or test regions during training.  This module solves
    the problem by grouping tiles at the *source-image level*: every tile
    from a given source ends up in exactly one partition.

    The source image is identified from the ``source_tif`` TIFF tag written
    by :mod:`src.vessels_detect.data.tiler` — no filename parsing is required.

**Class-Aware Assignment (replaces random shuffle)**
    A naive random shuffle of source images systematically violates target
    class ratios when annotations are unevenly distributed across source
    TIFs (e.g. most Double_hulled_Pirogue tiles coming from a single flight
    line).  The new algorithm uses a **deficit-weighted greedy assignment**:

    1.  Build a per-source annotation profile (class → count) by scanning
        ``labels_raw_dir`` *before* moving any files.
    2.  Process sources in descending order of priority-class count
        (Pirogue, Double_hulled_Pirogue by default) so the most
        class-constraining sources are assigned first.
    3.  For each source, score every eligible split by how much assigning
        that source would close the per-class deficit, weighted by class
        priority.  Assign to the highest-scoring split.
    4.  Capacity caps (derived from the requested ratios) prevent all
        sources from collapsing into train.

**Background Balancing**
    Satellite imagery of open water produces a large number of empty tiles.
    :class:`BackgroundBalancer` moves excess background tiles to an archive
    directory instead of deleting them, so they can be recovered if needed.
    The target background fraction is controlled by ``background_ratio``.

Typical usage::

    from src.vessels_detect.data.split import DatasetSplitter, SplitConfig
    from src.vessels_detect.data.split import BackgroundBalancer, BalanceConfig

    splitter = DatasetSplitter(SplitConfig(train_ratio=0.70, val_ratio=0.15))
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
from typing import Dict, List, Optional, Tuple

import rasterio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default priority: rarest / most important classes first.
# Class IDs follow the dataset.yaml mapping:
#   0 Pirogue  1 Double_hulled_Pirogue  2 Small_Motorboat
#   3 Medium_Motorboat  4 Large_Motorboat  5 Sailing_Boat
# ---------------------------------------------------------------------------
_DEFAULT_PRIORITY_CLASS_IDS: List[int] = [0, 1]
_DEFAULT_PRIORITY_WEIGHT:    float     = 4.0   # extra weight for priority classes


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Hyperparameters for :class:`DatasetSplitter`.

    Attributes:
        train_ratio: Fraction of *annotation instances* targeting the
            training set.  Assignment is performed at source-image
            granularity, so the achieved fraction will be approximate.
        val_ratio: Target fraction for the validation set.
        test_ratio: Target fraction for the test set.
            The three ratios must sum to 1.0 within floating-point
            tolerance.
        random_seed: Seed used when breaking ties between equally-scored
            splits and when shuffling background-only sources.
        priority_class_ids: Class IDs that receive extra weight in the
            deficit scoring function.  Defaults to [0, 1] (Pirogue and
            Double_hulled_Pirogue).
        priority_weight: Multiplicative weight applied to priority classes
            in the scoring function.  Values between 2 and 6 work well;
            higher = more aggressively balance those classes at the expense
            of the others.
    """

    train_ratio:        float     = 0.70
    val_ratio:          float     = 0.15
    test_ratio:         float     = 0.15
    random_seed:        int       = 42
    priority_class_ids: List[int] = field(
        default_factory=lambda: list(_DEFAULT_PRIORITY_CLASS_IDS)
    )
    priority_weight:    float     = _DEFAULT_PRIORITY_WEIGHT

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must equal 1.0, "
                f"got {total:.6f}."
            )
        if self.priority_weight < 1.0:
            raise ValueError(
                f"priority_weight must be ≥ 1.0, got {self.priority_weight}."
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
            val_ratio=cfg.get("val_ratio", 0.15),
            test_ratio=cfg.get("test_ratio", 0.15),
            random_seed=cfg.get("random_seed", 42),
            priority_class_ids=cfg.get(
                "priority_class_ids", list(_DEFAULT_PRIORITY_CLASS_IDS)
            ),
            priority_weight=cfg.get("priority_weight", _DEFAULT_PRIORITY_WEIGHT),
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
    random_seed:      int   = 42

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
# Internal helpers — I/O
# ---------------------------------------------------------------------------

def _read_source_tag(tile_path: Path) -> str:
    """Read the ``source_tif`` TIFF tag written by the tiler.

    Falls back to the tile stem when the tag is absent.

    Args:
        tile_path: Path to a GeoTIFF tile.

    Returns:
        The ``source_tif`` tag value, or the tile stem on failure.
    """
    try:
        with rasterio.open(tile_path) as ds:
            return ds.tags().get("source_tif", tile_path.stem)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cannot read TIFF tags from '%s': %s", tile_path.name, exc)
        return tile_path.stem


def _is_background(label_path: Path) -> bool:
    """Return ``True`` if a YOLO label file is empty (background tile).

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        ``True`` if the file is absent or contains no annotation lines.
    """
    if not label_path.exists():
        return True
    return label_path.read_text().strip() == ""


def _move_pair(
    src_img: Path,
    src_lbl: Optional[Path],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
) -> None:
    """Move an image tile and its label file to destination directories.

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
# Internal helpers — annotation profiling
# ---------------------------------------------------------------------------

def _build_source_profiles(
    source_to_tiles: Dict[str, List[Path]],
    labels_raw_dir: Path,
) -> Dict[str, Dict[int, int]]:
    """Scan label files and return per-source annotation class counts.

    This must be called *before* any files are moved so that labels are
    still in ``labels_raw_dir``.

    Args:
        source_to_tiles: Mapping ``{source_name: [tile_path, ...]}``.
        labels_raw_dir: Directory containing the raw YOLO ``.txt`` files.

    Returns:
        Mapping ``{source_name: {class_id: annotation_count}}``.
        Background-only sources have an empty inner dict.
    """
    profiles: Dict[str, Dict[int, int]] = {}
    for src, tiles in source_to_tiles.items():
        counts: Dict[int, int] = {}
        for tile_path in tiles:
            lbl = labels_raw_dir / f"{tile_path.stem}.txt"
            if not lbl.exists():
                continue
            for line in lbl.read_text().splitlines():
                line = line.strip()
                if line:
                    cid = int(line.split()[0])
                    counts[cid] = counts.get(cid, 0) + 1
        profiles[src] = counts
    return profiles


def _compute_split_capacities(
    n_sources: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    """Compute integer source-count caps that sum exactly to ``n_sources``.

    Capacities are set as the rounded ratio of total sources, with a
    minimum of 1 for any non-zero ratio.  The train split absorbs any
    rounding residual.

    Args:
        n_sources: Total number of unique source images.
        train_ratio: Target fraction for training.
        val_ratio: Target fraction for validation.
        test_ratio: Target fraction for testing.

    Returns:
        ``{"train": n_train, "val": n_val, "test": n_test}`` where
        ``n_train + n_val + n_test == n_sources``.
    """
    n_val  = max(1, round(n_sources * val_ratio))  if val_ratio  > 1e-9 else 0
    n_test = max(1, round(n_sources * test_ratio)) if test_ratio > 1e-9 else 0
    n_train = n_sources - n_val - n_test

    # Guard against over-allocation shrinking train below 1.
    if n_train < 1:
        # Trim val and test by 1 each until train ≥ 1.
        while n_train < 1 and (n_val > 0 or n_test > 0):
            if n_test > n_val:
                n_test -= 1
            else:
                n_val  -= 1
            n_train += 1

    return {"train": n_train, "val": n_val, "test": n_test}


# ---------------------------------------------------------------------------
# Internal helpers — class-aware assignment
# ---------------------------------------------------------------------------

def _deficit_score(
    profile:         Dict[int, int],
    split:           str,
    allocated:       Dict[str, Dict[int, int]],
    class_totals:    Dict[int, int],
    ratios:          Dict[str, float],
    priority_ids:    List[int],
    priority_weight: float,
) -> float:
    """Score the benefit of assigning *profile* to *split*.

    For every class ``c`` the source contributes ``delta_c`` annotations.
    The score is::

        Σ_c  w_c  *  delta_c  *  gap_c  /  total_c

    where:
        - ``w_c`` is ``priority_weight`` for priority classes, else ``1.0``
        - ``gap_c = target_frac - current_frac`` (positive ↔ split is
          under-represented in class c, so assigning more is beneficial;
          negative ↔ already over-represented, penalised)
        - Normalising by ``total_c`` makes all classes contribute on the
          same relative scale regardless of absolute frequency.

    Args:
        profile:         ``{class_id: count}`` for the source being scored.
        split:           Candidate split name (``"train"``, ``"val"``, ``"test"``).
        allocated:       Current accumulated counts per split per class.
        class_totals:    Global annotation count per class.
        ratios:          Target ratio per split.
        priority_ids:    Class IDs that receive extra scoring weight.
        priority_weight: Multiplier for priority class contributions.

    Returns:
        A float score; higher is better.
    """
    target_frac = ratios[split]
    score = 0.0
    for cid, delta in profile.items():
        if delta == 0:
            continue
        total = class_totals.get(cid, 1)
        current_frac = allocated[split].get(cid, 0) / total
        gap  = target_frac - current_frac
        w    = priority_weight if cid in priority_ids else 1.0
        score += w * delta * gap / total
    return score


def _assign_splits_class_aware(
    source_images:      List[str],
    source_profiles:    Dict[str, Dict[int, int]],
    capacities:         Dict[str, int],
    train_ratio:        float,
    val_ratio:          float,
    test_ratio:         float,
    priority_class_ids: List[int],
    priority_weight:    float,
    rng:                random.Random,
) -> Dict[str, str]:
    """Assign source images to splits using deficit-weighted greedy search.

    Processing order
    ~~~~~~~~~~~~~~~~
    1.  Sources with at least one priority-class annotation, sorted by
        descending priority-class count (most constrained first).
    2.  Sources with other annotations only, sorted by descending total
        annotation count.
    3.  Background-only sources (shuffled randomly — class balance is
        irrelevant for them).

    Assignment rule
    ~~~~~~~~~~~~~~~
    For each source, pick the eligible split (not yet at capacity) with the
    highest :func:`_deficit_score`.  Ties are broken randomly.

    Args:
        source_images:      All unique source image names.
        source_profiles:    ``{source: {class_id: count}}`` from
                            :func:`_build_source_profiles`.
        capacities:         Maximum number of sources per split from
                            :func:`_compute_split_capacities`.
        train_ratio:        Target train fraction.
        val_ratio:          Target val fraction.
        test_ratio:         Target test fraction.
        priority_class_ids: Class IDs to weight heavily.
        priority_weight:    Weight multiplier for priority classes.
        rng:                Seeded random instance for tie-breaking.

    Returns:
        ``{source_name: split_name}`` assignment dictionary.
    """
    splits  = ["train", "val", "test"]
    ratios  = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    # ── Global class totals ───────────────────────────────────────────────
    class_totals: Dict[int, int] = {}
    for profile in source_profiles.values():
        for cid, cnt in profile.items():
            class_totals[cid] = class_totals.get(cid, 0) + cnt

    all_class_ids = sorted(class_totals.keys())

    # ── Current allocation tracker ────────────────────────────────────────
    allocated: Dict[str, Dict[int, int]] = {
        sp: {cid: 0 for cid in all_class_ids} for sp in splits
    }
    split_used: Dict[str, int] = {sp: 0 for sp in splits}

    # ── Sort sources into three priority tiers ────────────────────────────
    def _priority_count(src: str) -> int:
        return sum(
            source_profiles[src].get(cid, 0) for cid in priority_class_ids
        )

    def _total_count(src: str) -> int:
        return sum(source_profiles[src].values())

    priority_annotated = [
        s for s in source_images if _priority_count(s) > 0
    ]
    other_annotated = [
        s for s in source_images
        if _total_count(s) > 0 and _priority_count(s) == 0
    ]
    background_only = [
        s for s in source_images if _total_count(s) == 0
    ]

    priority_annotated.sort(key=_priority_count, reverse=True)
    other_annotated.sort(key=_total_count, reverse=True)
    rng.shuffle(background_only)

    ordered_sources = priority_annotated + other_annotated + background_only

    logger.debug(
        "Assignment order: %d priority-annotated + %d other-annotated + "
        "%d background-only  |  capacities: %s",
        len(priority_annotated), len(other_annotated), len(background_only),
        capacities,
    )

    # ── Greedy assignment ─────────────────────────────────────────────────
    assignment: Dict[str, str] = {}

    for src in ordered_sources:
        profile = source_profiles.get(src, {})

        # Eligible splits: those that still have capacity.
        eligible = [
            sp for sp in splits
            if split_used[sp] < capacities[sp]
        ]

        # Safety net: if somehow no split has capacity left (rounding edge
        # case), fall back to the least-full split.
        if not eligible:
            logger.warning(
                "All splits at capacity while assigning '%s'. "
                "Falling back to least-full split.", src
            )
            eligible = [min(splits, key=lambda sp: split_used[sp])]

        # Score each eligible split.
        scores = {
            sp: _deficit_score(
                profile, sp, allocated,
                class_totals, ratios,
                priority_class_ids, priority_weight,
            )
            for sp in eligible
        }

        # Pick the highest-scoring split; break ties randomly.
        max_score = max(scores.values())
        best_splits = [sp for sp, sc in scores.items() if abs(sc - max_score) < 1e-12]
        chosen = rng.choice(best_splits)

        assignment[src] = chosen
        split_used[chosen] += 1
        for cid, cnt in profile.items():
            allocated[chosen][cid] = allocated[chosen].get(cid, 0) + cnt

    return assignment


# ---------------------------------------------------------------------------
# Internal helpers — diagnostics
# ---------------------------------------------------------------------------

def _log_class_distribution(
    assignment:      Dict[str, str],
    source_profiles: Dict[str, Dict[int, int]],
    class_names:     Optional[Dict[int, str]] = None,
) -> None:
    """Log a per-class annotation count table across splits.

    Args:
        assignment:      ``{source_name: split_name}``.
        source_profiles: ``{source_name: {class_id: count}}``.
        class_names:     Optional ``{class_id: name}`` for readable output.
    """
    splits      = ["train", "val", "test"]
    tally:      Dict[str, Dict[int, int]] = {sp: {} for sp in splits}
    all_class_ids: set = set()

    for src, sp in assignment.items():
        for cid, cnt in source_profiles.get(src, {}).items():
            tally[sp][cid] = tally[sp].get(cid, 0) + cnt
            all_class_ids.add(cid)

    if not all_class_ids:
        logger.info("  (no annotated tiles found — nothing to report)")
        return

    header = f"  {'Class':<28}  {'Train':>7}  {'Val':>7}  {'Test':>7}  {'Total':>7}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))

    for cid in sorted(all_class_ids):
        label = (class_names or {}).get(cid, f"class_{cid}")
        tr = tally["train"].get(cid, 0)
        vl = tally["val"].get(cid, 0)
        te = tally["test"].get(cid, 0)
        tot = tr + vl + te
        logger.info(
            "  %-28s  %7d  %7d  %7d  %7d   "
            "(%.1f%% / %.1f%% / %.1f%%)",
            label, tr, vl, te, tot,
            100 * tr / (tot or 1),
            100 * vl / (tot or 1),
            100 * te / (tot or 1),
        )


# ---------------------------------------------------------------------------
# Public API — splitting
# ---------------------------------------------------------------------------

class DatasetSplitter:
    """Assigns GeoTIFF tiles to train / val / test partitions.

    Grouping is performed at the *source-image level* to guarantee zero
    spatial leakage: tiles that share an origin image always end up in the
    same partition.

    Assignment is driven by a **deficit-weighted greedy algorithm** that
    minimises per-class ratio deviation, with configurable extra weight for
    priority classes (Pirogue, Double_hulled_Pirogue by default).

    Args:
        config: :class:`SplitConfig` specifying ratios, seed, and class
            priorities.
        class_names: Optional ``{class_id: name}`` mapping used purely for
            the diagnostic log table.  Has no effect on assignment logic.
    """

    def __init__(
        self,
        config:      SplitConfig,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self._cfg         = config
        self._class_names = class_names

    def split(
        self,
        tiles_dir:      Path,
        labels_raw_dir: Path,
        images_dir:     Path,
        labels_dir:     Path,
    ) -> Dict[str, List[Path]]:
        """Distribute tiles and labels into split subdirectories.

        The source image for each tile is identified by the ``source_tif``
        TIFF tag.  All tiles sharing a source are assigned to the same
        split, then physically moved from *tiles_dir* / *labels_raw_dir*
        into *images_dir/<split>/* and *labels_dir/<split>/*.

        Assignment maximises class-ratio fidelity using a deficit-weighted
        greedy algorithm; priority classes receive extra scoring weight.

        Args:
            tiles_dir:      Directory of GeoTIFF tiles (tiler output).
            labels_raw_dir: Directory of YOLO ``.txt`` files (annotator
                            output, flat — not yet split into sub-dirs).
            images_dir:     Root output directory; ``train/``, ``val/``,
                            and ``test/`` subdirectories are created here.
            labels_dir:     Root output directory for labels; mirrors the
                            structure of *images_dir*.

        Returns:
            ``{split_name: [moved_tile_path, ...]}`` listing every tile
            moved into each split.

        Raises:
            FileNotFoundError: If *tiles_dir* does not exist.
            RuntimeError: If no ``.tif`` tiles are found in *tiles_dir*.
        """
        if not tiles_dir.exists():
            raise FileNotFoundError(f"tiles_dir does not exist: {tiles_dir}")

        tile_paths = sorted(tiles_dir.glob("*.tif"))
        if not tile_paths:
            raise RuntimeError(f"No .tif tiles found in '{tiles_dir}'.")

        cfg = self._cfg

        # ── 1. Group tiles by source image ────────────────────────────────
        source_to_tiles: Dict[str, List[Path]] = {}
        for tp in tile_paths:
            src = _read_source_tag(tp)
            source_to_tiles.setdefault(src, []).append(tp)

        source_images = sorted(source_to_tiles.keys())
        n_sources     = len(source_images)

        logger.info(
            "Split: %d tile(s) from %d source image(s).  "
            "Target ratios: train=%.2f  val=%.2f  test=%.2f",
            len(tile_paths), n_sources,
            cfg.train_ratio, cfg.val_ratio, cfg.test_ratio,
        )

        # ── 2. Build annotation profiles (must happen before any file moves) ──
        logger.info("Building per-source annotation profiles …")
        source_profiles = _build_source_profiles(source_to_tiles, labels_raw_dir)

        n_annotated_sources = sum(
            1 for p in source_profiles.values() if p
        )
        logger.info(
            "  %d / %d source image(s) carry at least one annotation.",
            n_annotated_sources, n_sources,
        )

        # ── 3. Compute integer capacity caps per split ─────────────────────
        capacities = _compute_split_capacities(
            n_sources, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio
        )
        logger.info("  Source-image capacities: %s", capacities)

        # ── 4. Class-aware greedy assignment ──────────────────────────────
        rng = random.Random(cfg.random_seed)
        assignment = _assign_splits_class_aware(
            source_images=source_images,
            source_profiles=source_profiles,
            capacities=capacities,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            priority_class_ids=cfg.priority_class_ids,
            priority_weight=cfg.priority_weight,
            rng=rng,
        )

        # ── 5. Log per-class distribution ─────────────────────────────────
        logger.info("Annotation distribution after assignment:")
        _log_class_distribution(assignment, source_profiles, self._class_names)

        # ── 6. Move tiles and labels to split directories ──────────────────
        split_counts: Dict[str, int]      = {"train": 0, "val": 0, "test": 0}
        result:       Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

        for src_name, tiles in source_to_tiles.items():
            split_name = assignment[src_name]
            dst_img    = images_dir / split_name
            dst_lbl    = labels_dir / split_name

            for tile_path in tiles:
                label_path = labels_raw_dir / f"{tile_path.stem}.txt"
                _move_pair(
                    tile_path,
                    label_path if label_path.exists() else None,
                    dst_img,
                    dst_lbl,
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
        images_dir:  Path,
        labels_dir:  Path,
        archive_dir: Path,
        splits:      List[str],
    ) -> None:
        """Balance background tiles for the specified splits.

        For each split listed in *splits*, background tiles in excess of
        the ``background_ratio`` target are moved to::

            <archive_dir>/images/<split>/
            <archive_dir>/labels/<split>/

        Args:
            images_dir:  Root directory containing ``<split>/``
                         subdirectories with GeoTIFF tiles.
            labels_dir:  Root directory containing ``<split>/``
                         subdirectories with YOLO ``.txt`` label files.
            archive_dir: Root directory for archived excess tiles.
            splits:      List of split names to process
                         (e.g., ``["train", "val"]``).
        """
        ratio = self._cfg.background_ratio
        rng   = random.Random(self._cfg.random_seed)

        logger.info(
            "Background balancing: target ratio=%.2f  splits=%s",
            ratio, splits,
        )

        for split in splits:
            self._balance_split(
                split, images_dir, labels_dir, archive_dir, ratio, rng
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _balance_split(
        self,
        split:       str,
        images_dir:  Path,
        labels_dir:  Path,
        archive_dir: Path,
        ratio:       float,
        rng:         random.Random,
    ) -> None:
        """Balance background tiles in one split.

        Args:
            split:       Split name (e.g., ``"train"``).
            images_dir:  Root images directory (parent of ``<split>/``).
            labels_dir:  Root labels directory (parent of ``<split>/``).
            archive_dir: Root archive directory.
            ratio:       Target background fraction.
            rng:         Seeded :class:`random.Random` for reproducibility.
        """
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split

        if not img_split_dir.exists():
            logger.warning(
                "Balance: images/%s/ does not exist — skipping.", split
            )
            return

        annotated:  List[str] = []
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

        # Desired bg count: bg = ann × ratio / (1 − ratio)
        target_bg = int(n_ann * ratio / (1.0 - ratio))
        target_bg = min(target_bg, n_bg)

        rng.shuffle(background)
        move_bg = background[target_bg:]

        n_total_new   = n_ann + target_bg
        actual_bg_pct = target_bg / n_total_new * 100.0 if n_total_new > 0 else 0.0

        logger.info(
            "  [%s]  annotated=%d  bg_before=%d  bg_kept=%d (%.1f%%)"
            "  bg_archived=%d  new_total=%d",
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