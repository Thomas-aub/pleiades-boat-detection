#!/usr/bin/env python3
"""
scripts/grid_search.py
----------------------
Automated grid search over the maritime micro-object detection pipeline.

Grid space  (3 × 3 × 2 × 2 = 36 runs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    spatial upsampling : [1.0, 2.0, 4.0]   (cubic interpolation always)
    tile / imgsz       : [640, 1024, 1536]
    class config       : [A — 6 classes | B — 4 classes]
    architecture       : [standard *.pt  | custom P2 *.yaml]

Preprocessing cache strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To avoid redundant I/O, preprocessing artefacts are grouped by their
minimal differentiating key:

    (upscale_ratio, class_config)
        → stages 2–4 (spatial · annotations · split) — **computed once**

    (upscale_ratio, tile_size, class_config)
        → stage 5 (tiling) — **computed once**

Training (architecture-only difference) always runs for all 36 runs.

Class mapping strategies
~~~~~~~~~~~~~~~~~~~~~~~~
Config A — 6 classes  : original taxonomy; skip classes 9, 10, 11.
Config B — 4 classes  : merged taxonomy designed to reduce inter-class
    ambiguity in visually similar categories:
        0  Pirogue             ← original  0, 8, 10
        1  Double_hulled       ← original  1
        2  Small_Motorboat     ← original  2, 5 (Sailing_Boat)
        3  Large_Motorboat     ← original  3, 4, 6
    skip: 9, 11.
    The class_map is injected into ``preprocessing.yaml`` at Stage 3
    (annotations), so all downstream label files already carry remapped
    IDs.  :func:`remap_label_files` is provided as a standalone utility
    for post-hoc remapping without re-running the pipeline.

Fault tolerance
~~~~~~~~~~~~~~~
CUDA OOM errors are detected by scanning combined stdout + stderr for
known markers.  Any failing run is logged and the grid search continues
with the next combination.  Runs that depend on a previously-failed
preprocessing key are immediately marked as failed without retrying.

Output layout
~~~~~~~~~~~~~
::

    data/grid_search/<preproc_key>/        — stages 2–4 artefacts
    data/grid_search/<tiling_key>/tiled/   — stage 5 tile GeoTIFFs
    data/grid_search/<tiling_key>/dataset.yaml
    configs/grid_search/preproc_<key>.yaml — auto-generated temp configs
    configs/grid_search/tiling_<key>.yaml
    configs/grid_search/train_<run_id>.yaml
    logs/grid_search/<run_id>_<stage>.log  — per-run subprocess capture
    logs/grid_search_summary.txt           — final results table
    boat_obb/gridsearch/<run_id>/          — YOLO training outputs

Usage::

    PYTHONPATH=. python scripts/grid_search.py
    PYTHONPATH=. python scripts/grid_search.py --log-level DEBUG
    PYTHONPATH=. python scripts/grid_search.py --dry-run
    PYTHONPATH=. python scripts/grid_search.py \\
        --preproc-config configs/my_preproc.yaml \\
        --train-config   configs/my_train.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Dict, List, Optional, Set, Tuple

import yaml

# =============================================================================
# Project-level constants
# =============================================================================

_PYTHON = sys.executable                           # honours active venv / conda
_ROOT   = Path(__file__).resolve().parent.parent   # project root

_GS_DATA_ROOT    = _ROOT / "data"    / "grid_search"  # preprocessing artefacts
_GS_CONFIGS_ROOT = _ROOT / "configs" / "grid_search"  # temp YAML files
_GS_LOGS_ROOT    = _ROOT / "logs"    / "grid_search"  # captured subprocess output

# Fixed training overrides enforced on every run (A40-optimised quick eval).
# These are applied last in _patch_train_config so they always win.
_TRAIN_OVERRIDES: Dict[str, dict] = {
    "training": {
        "epochs":     50,
        "patience":   15,
        "batch_size": 16,
        "workers":    8,
    },
    "augmentation": {
        "scale":        0.15,
        "close_mosaic": 0,
    },
}

# Checked case-insensitively in combined stdout + stderr of every subprocess.
_OOM_MARKERS: Tuple[str, ...] = (
    "cuda out of memory",
    "outofmemoryerror",
    "cuda error: out of memory",
    "torch.cuda.outofmemoryerror",
)

logger = logging.getLogger(__name__)


# =============================================================================
# Grid space
# =============================================================================

UPSCALE_RATIOS: List[float] = [1.0, 2.0, 4.0]
TILE_SIZES:     List[int]   = [640, 1024, 1536]


@dataclass(frozen=True)
class ClassConfig:
    """One annotation class-mapping strategy injected into Stage 3 (annotations)."""

    config_id:    str              # "A" or "B" — used in run IDs / directory names
    n_classes:    int              # number of YOLO output classes
    class_map:    Dict[int, int]   # GeoJSON class_id → YOLO class_id
    skip_classes: Tuple[int, ...]  # GeoJSON class IDs discarded at annotation time
    class_names:  Tuple[str, ...]  # ordered YOLO class names (index = YOLO class_id)


# ── Config A: original 6-class taxonomy ──────────────────────────────────────
# Classes 9, 10, and 11 are discarded; all other mappings are identity or
# the existing merge of class 6 → Large_Motorboat.
#
# ── Config B: merged 4-class taxonomy ────────────────────────────────────────
# Merges visually similar categories to reduce inter-class confusion:
#   0  Pirogue             ← original 0, 8, 10  (all small open boats)
#   1  Double_hulled       ← original 1
#   2  Small_Motorboat     ← original 2, 5       (Sailing_Boat merged in)
#   3  Large_Motorboat     ← original 3, 4, 6    (motorboat size collapse)
CLASS_CONFIGS: Dict[str, ClassConfig] = {
    "A": ClassConfig(
        config_id="A",
        n_classes=6,
        class_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 8: 0, 10: 0},
        skip_classes=(9, 11),
        class_names=(
            "Pirogue", "Double_hulled_Pirogue", "Small_Motorboat",
            "Medium_Motorboat", "Large_Motorboat", "Sailing_Boat",
        ),
    ),
    "B": ClassConfig(
        config_id="B",
        n_classes=4,
        class_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 3, 8: 0, 10: 0},
        skip_classes=(9, 11),
        class_names=(
            "Pirogue", "Double_hulled_Pirogue", "Small_Motorboat", "Large_Motorboat",
        ),
    ),
}


@dataclass(frozen=True)
class ArchConfig:
    """One model architecture variant."""

    arch_id:    str   # "std" or "p2"
    weights:    str   # model.weights   (*.pt → direct load | *.yaml → custom arch)
    pretrained: str   # model.pretrained (empty string = no partial weight transfer)


# ── std: direct fine-tuning from the pretrained checkpoint ───────────────────
# ── p2:  custom architecture YAML + partial weight transfer from base *.pt ───
ARCH_CONFIGS: Dict[str, ArchConfig] = {
    "std": ArchConfig(
        arch_id="std",
        weights="weights/yolo26m-obb.pt",
        pretrained="",
    ),
    "p2": ArchConfig(
        arch_id="p2",
        weights="configs/yolo26m-obb-p2.yaml",
        pretrained="weights/yolo26m-obb.pt",
    ),
}


def _fmt_scale(v: float) -> str:
    """Compact string for an upscale ratio: ``1.0`` → ``"1"``, ``2.5`` → ``"2.5"``."""
    return str(int(v)) if v == int(v) else str(v)


@dataclass
class GridRun:
    """A fully-specified point in the 36-run search grid."""

    upscale_ratio: float
    tile_size:     int
    class_cfg:     ClassConfig
    arch_cfg:      ArchConfig

    # ------------------------------------------------------------------
    # Derived identifiers
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """Unique human-readable identifier, e.g. ``up2_t1024_cl4_p2``.

        Format: ``up{scale}_t{tile}_cl{n_classes}_{arch_id}``
        """
        return (
            f"up{_fmt_scale(self.upscale_ratio)}"
            f"_t{self.tile_size}"
            f"_cl{self.class_cfg.n_classes}"
            f"_{self.arch_cfg.arch_id}"
        )

    @property
    def preproc_key(self) -> str:
        """Cache key for stages 2–4.

        Shared by all runs with the same upsampling ratio and class config,
        regardless of tile size or architecture.
        """
        return f"up{_fmt_scale(self.upscale_ratio)}_cl{self.class_cfg.n_classes}"

    @property
    def tiling_key(self) -> str:
        """Cache key for stage 5 (tiling).

        Shared by all runs with the same upsampling, tile size, and class
        config, regardless of architecture.
        """
        return (
            f"up{_fmt_scale(self.upscale_ratio)}"
            f"_t{self.tile_size}"
            f"_cl{self.class_cfg.n_classes}"
        )


def build_grid() -> List[GridRun]:
    """Enumerate all 36 grid search runs in a deterministic order.

    Outer-to-inner iteration: upscale → tile_size → class_config → arch.
    This ordering groups related preprocessing keys together, maximising
    the benefit of the preprocessing cache.

    Returns:
        Ordered list of :class:`GridRun` descriptors.
    """
    runs: List[GridRun] = []
    for upscale in UPSCALE_RATIOS:
        for tile_size in TILE_SIZES:
            for cl_id in ("A", "B"):
                for arch_id in ("std", "p2"):
                    runs.append(GridRun(
                        upscale_ratio=upscale,
                        tile_size=tile_size,
                        class_cfg=CLASS_CONFIGS[cl_id],
                        arch_cfg=ARCH_CONFIGS[arch_id],
                    ))
    return runs


# =============================================================================
# Path resolution
# =============================================================================

def _preproc_dirs(run: GridRun) -> Dict[str, Path]:
    """Resolve stage 2–4 output paths for *run*'s preprocessing key.

    All three paths share the same parent directory, keyed on
    ``(upscale_ratio, class_config)``.

    Args:
        run: Grid run descriptor.

    Returns:
        Dict with keys ``spatial_dir``, ``labels_dir``, ``dataset_dir``.
    """
    base = _GS_DATA_ROOT / run.preproc_key
    return {
        "spatial_dir":  base / "spatial",
        "labels_dir":   base / "labels",
        "dataset_dir":  base / "dataset",
    }


def _tiling_root(run: GridRun) -> Path:
    """Resolve the tiled-dataset root directory for *run*'s tiling key."""
    return _GS_DATA_ROOT / run.tiling_key / "tiled"


def _dataset_yaml_path(run: GridRun) -> Path:
    """Resolve the YOLO ``dataset.yaml`` path for *run*."""
    return _GS_DATA_ROOT / run.tiling_key / "dataset.yaml"


def _preproc_config_path(key: str) -> Path:
    return _GS_CONFIGS_ROOT / f"preproc_{key}.yaml"


def _tiling_config_path(key: str) -> Path:
    return _GS_CONFIGS_ROOT / f"tiling_{key}.yaml"


def _train_config_path(run_id: str) -> Path:
    return _GS_CONFIGS_ROOT / f"train_{run_id}.yaml"


def _stage_log_path(run_id: str, stage: str) -> Path:
    return _GS_LOGS_ROOT / f"{run_id}_{stage}.log"


# =============================================================================
# YAML patching
# =============================================================================

def _patch_preproc_stages_2_to_4(base_cfg: dict, run: GridRun) -> dict:
    """Build a ``preprocessing.yaml`` with only stages 2–4 enabled.

    Overrides ``spatial_dir``, ``labels_dir``, and ``dataset_dir`` to the
    run's preprocessing cache directory.  ``raw_dir`` and
    ``radiometric_dir`` are kept from the base config (stage 1 is assumed
    to have been run already).

    Args:
        base_cfg: Raw dict loaded from ``configs/preprocessing.yaml``.
        run:      Grid run descriptor.

    Returns:
        Patched config dict, ready for :func:`_write_yaml`.
    """
    cfg  = copy.deepcopy(base_cfg)
    dirs = _preproc_dirs(run)

    # Keep raw_dir and radiometric_dir from the base config verbatim.
    cfg["paths"]["spatial_dir"]  = str(dirs["spatial_dir"])
    cfg["paths"]["labels_dir"]   = str(dirs["labels_dir"])
    cfg["paths"]["dataset_dir"]  = str(dirs["dataset_dir"])

    cfg.setdefault("spatial", {})
    cfg["spatial"]["upscale_ratio"] = run.upscale_ratio
    cfg["spatial"]["interpolation"] = "cubic"   # enforced for all runs

    cfg.setdefault("annotations", {})
    cfg["annotations"]["class_map"]    = dict(run.class_cfg.class_map)
    cfg["annotations"]["skip_classes"] = list(run.class_cfg.skip_classes)

    # Enable exactly stages 2, 3, 4; disable everything else.
    _set_enabled_stages(cfg, {"spatial", "annotations", "split"})

    return cfg


def _patch_preproc_stage_5(base_cfg: dict, run: GridRun) -> dict:
    """Build a ``preprocessing.yaml`` with only stage 5 (tiling) enabled.

    Points ``dataset_dir`` to the stage 2–4 cache for this run's preproc
    key and ``tiled_dir`` to its own tiling cache.

    Args:
        base_cfg: Raw dict loaded from ``configs/preprocessing.yaml``.
        run:      Grid run descriptor.

    Returns:
        Patched config dict, ready for :func:`_write_yaml`.
    """
    cfg  = copy.deepcopy(base_cfg)
    dirs = _preproc_dirs(run)

    cfg["paths"]["dataset_dir"] = str(dirs["dataset_dir"])
    cfg["paths"]["tiled_dir"]   = str(_tiling_root(run))

    cfg.setdefault("tiling", {})
    cfg["tiling"]["tile_size"] = run.tile_size   # stride = tile_size - overlap

    _set_enabled_stages(cfg, {"tiling"})

    return cfg


def _patch_train_config(base_cfg: dict, run: GridRun) -> dict:
    """Build a ``train.yaml`` for this grid run.

    Applies model variant, dataset path, image size, run name, and the
    fixed A40 overrides (which always take the highest precedence).

    Args:
        base_cfg: Raw dict loaded from ``configs/train.yaml``.
        run:      Grid run descriptor.

    Returns:
        Patched config dict, ready for :func:`_write_yaml`.
    """
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("model", {})
    cfg["model"]["weights"]    = run.arch_cfg.weights
    cfg["model"]["pretrained"] = run.arch_cfg.pretrained

    cfg.setdefault("training", {})
    cfg["training"]["imgsz"]        = run.tile_size
    cfg["training"]["dataset_yaml"] = str(_dataset_yaml_path(run))
    cfg["training"]["run_name"]     = f"gridsearch/{run.run_id}"
    cfg["training"]["resume"]       = False

    # Fixed A40 overrides — applied last to guarantee they are never shadowed.
    for section, overrides in _TRAIN_OVERRIDES.items():
        cfg.setdefault(section, {})
        cfg[section].update(overrides)

    return cfg


def _set_enabled_stages(cfg: dict, active: Set[str]) -> None:
    """Mutate *cfg* so only the named stages have ``enabled: true``."""
    for stage in cfg.get("stages", []):
        stage["enabled"] = stage.get("name") in active


# =============================================================================
# Dataset YAML writer
# =============================================================================

def write_dataset_yaml(run: GridRun) -> Path:
    """Write a YOLO-compatible ``dataset.yaml`` for *run*'s tiled output.

    The YAML is placed next to the tiled output directory so that all
    artefacts for a given tiling key are co-located.

    Args:
        run: Grid run descriptor.

    Returns:
        Path to the written ``dataset.yaml``.
    """
    yaml_path = _dataset_yaml_path(run)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    content = {
        # Ultralytics resolves train/val paths relative to ``path``.
        "path":  str(_tiling_root(run)),
        "train": "images/train",
        "val":   "images/val",
        "nc":    run.class_cfg.n_classes,
        "names": list(run.class_cfg.class_names),
        # Provenance field — ignored by YOLO, useful for debugging.
        "_gs_tiling_key": run.tiling_key,
    }

    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.dump(content, fh, default_flow_style=False, sort_keys=False)

    logger.debug("dataset.yaml → %s", yaml_path)
    return yaml_path


# =============================================================================
# Label remapping utility (standalone — not called by the grid search loop)
# =============================================================================

def remap_label_files(
    label_dir: Path,
    class_map: Dict[int, int],
    skip_classes: Collection[int],
    *,
    in_place: bool = True,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Remap YOLO OBB class IDs across all ``.txt`` label files in a directory.

    **This function is not called by the grid search loop.**  In the normal
    flow, class remapping is handled upstream at Stage 3 (annotations) via
    ``preprocessing.yaml``'s ``class_map`` / ``skip_classes`` fields.  This
    utility is provided for post-hoc remapping — e.g. when exploring a new
    class taxonomy without re-running the full preprocessing pipeline.

    Each label line has the YOLO OBB format::

        class_id  x1 y1 x2 y2 x3 y3 x4 y4   (all normalised to [0, 1])

    Lines whose ``class_id`` is in *skip_classes* are dropped.
    Lines whose ``class_id`` appears in *class_map* have their ID replaced.
    Lines with IDs absent from both are kept unchanged (a warning is emitted).

    Args:
        label_dir:    Directory containing YOLO OBB ``.txt`` label files.
        class_map:    ``{original_class_id: new_class_id}`` remapping table.
        skip_classes: Class IDs to discard entirely.
        in_place:     Overwrite each file when ``True``; write output to
                      ``label_dir / "remapped/"`` when ``False``.
        dry_run:      Log planned actions without writing any files.

    Returns:
        ``(n_files, n_annotations)`` — counts of processed files and
        annotation lines retained after remapping.

    Raises:
        FileNotFoundError: If *label_dir* does not exist.
    """
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    skip_set = set(skip_classes)
    out_dir  = label_dir if in_place else label_dir / "remapped"
    if not in_place and not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_annots = 0

    for txt_path in sorted(label_dir.glob("*.txt")):
        raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
        out_lines: List[str] = []

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue

            parts    = line.split()
            orig_cls = int(parts[0])

            if orig_cls in skip_set:
                continue

            new_cls = class_map.get(orig_cls)
            if new_cls is None:
                logger.warning(
                    "remap_label_files: unknown class_id=%d in '%s' — kept as-is.",
                    orig_cls, txt_path.name,
                )
                new_cls = orig_cls

            out_lines.append(f"{new_cls} " + " ".join(parts[1:]))
            n_annots += 1

        if not dry_run:
            (out_dir / txt_path.name).write_text(
                "\n".join(out_lines), encoding="utf-8"
            )
        n_files += 1

    logger.info(
        "remap_label_files: %d file(s), %d annotation(s)  %s → %s",
        n_files, n_annots, label_dir, out_dir,
    )
    return n_files, n_annots


# =============================================================================
# Subprocess runner
# =============================================================================

def _is_oom(text: str) -> bool:
    """Return ``True`` if *text* contains a CUDA out-of-memory marker."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in _OOM_MARKERS)


def _run_stage(
    cmd: List[str],
    run_id: str,
    stage_label: str,
    log_path: Path,
) -> Tuple[int, bool]:
    """Run *cmd* via ``subprocess.run``, capturing output to *log_path*.

    stdout and stderr are merged, written to *log_path*, and the tail is
    emitted through the Python logger on failure.  For long-running training
    jobs, real-time progress can be monitored with ``tail -f <log_path>``.

    Args:
        cmd:         Full command list (Python strings).
        run_id:      Human-readable run identifier used in log messages.
        stage_label: Short label for this stage, e.g. ``"training"``.
        log_path:    Destination file for the full captured output.

    Returns:
        ``(returncode, is_oom)`` — ``is_oom`` is ``True`` when CUDA OOM
        markers are detected anywhere in the subprocess output.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info("[%s] %s — %s", run_id, stage_label, cmd_str)

    t0 = time.perf_counter()
    result = subprocess.run(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout for a single stream
        text=True,
        cwd=str(_ROOT),
    )
    elapsed = time.perf_counter() - t0
    oom     = _is_oom(result.stdout)

    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# cmd      : {cmd_str}\n")
        fh.write(f"# exit code: {result.returncode}\n")
        fh.write(f"# elapsed  : {elapsed:.1f}s\n")
        fh.write("# " + "=" * 68 + "\n")
        fh.write(result.stdout)

    if result.returncode != 0:
        tail = result.stdout[-4096:]   # last ~4 KB is almost always diagnostic
        if oom:
            logger.warning(
                "[%s] %s — CUDA OOM after %.1fs.  Full log: %s",
                run_id, stage_label, elapsed, log_path,
            )
        else:
            logger.error(
                "[%s] %s — FAILED (rc=%d, %.1fs)\n%s\nFull log: %s",
                run_id, stage_label, result.returncode, elapsed, tail, log_path,
            )
    else:
        logger.info(
            "[%s] %s — done in %.1fs.  Log: %s",
            run_id, stage_label, elapsed, log_path,
        )

    return result.returncode, oom


# =============================================================================
# Run result
# =============================================================================

@dataclass
class RunResult:
    """Outcome of a single grid search run."""

    run_id:       str
    status:       str            # "success" | "oom" | "failed" | "skipped"
    failed_stage: Optional[str] = None   # first stage that caused the failure
    elapsed_s:    float         = 0.0


# =============================================================================
# Grid search orchestrator
# =============================================================================

def run_grid_search(
    runs: List[GridRun],
    base_preproc_cfg: dict,
    base_train_cfg: dict,
    dry_run: bool = False,
) -> List[RunResult]:
    """Execute all grid search runs.

    Preprocessing stages 2–4 and stage 5 are de-duplicated across runs that
    share the same preprocessing or tiling key.  If a shared preprocessing
    stage fails, all subsequent runs that depend on that key are immediately
    marked as failed without retrying.

    Args:
        runs:             Ordered list of :class:`GridRun` descriptors.
        base_preproc_cfg: Raw dict from ``configs/preprocessing.yaml``.
        base_train_cfg:   Raw dict from ``configs/train.yaml``.
        dry_run:          Write all YAML configs but skip subprocess calls.

    Returns:
        List of :class:`RunResult` in the same order as *runs*.
    """
    _GS_CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)

    completed_preproc: Set[str] = set()
    completed_tiling:  Set[str] = set()
    failed_preproc:    Set[str] = set()   # keys whose stage 2–4 failed
    failed_tiling:     Set[str] = set()   # keys whose stage 5 failed
    results: List[RunResult]    = []

    total = len(runs)

    for idx, run in enumerate(runs, 1):
        logger.info("")
        logger.info("=" * 70)
        logger.info("RUN %d / %d  —  %s", idx, total, run.run_id)
        logger.info("=" * 70)

        t_run = time.perf_counter()
        res   = RunResult(run_id=run.run_id, status="failed")

        # ------------------------------------------------------------------
        # Stages 2–4 : spatial · annotations · split
        # ------------------------------------------------------------------
        if run.preproc_key in failed_preproc:
            logger.warning(
                "[%s] Skipped — preprocessing key '%s' failed in an earlier run.",
                run.run_id, run.preproc_key,
            )
            res.status       = "failed"
            res.failed_stage = "preprocessing (inherited)"
            res.elapsed_s    = time.perf_counter() - t_run
            results.append(res)
            continue

        if run.preproc_key not in completed_preproc:
            preproc_cfg  = _patch_preproc_stages_2_to_4(base_preproc_cfg, run)
            preproc_yaml = _preproc_config_path(run.preproc_key)
            _write_yaml(preproc_cfg, preproc_yaml)
            logger.info("[%s] Preprocessing config → %s", run.run_id, preproc_yaml)

            if not dry_run:
                rc, oom = _run_stage(
                    cmd=[_PYTHON, "scripts/preprocessing.py",
                         "--config", str(preproc_yaml)],
                    run_id=run.run_id,
                    stage_label="preprocessing (stages 2–4)",
                    log_path=_stage_log_path(run.run_id, "preproc"),
                )
                if rc != 0:
                    failed_preproc.add(run.preproc_key)
                    res.status       = "oom" if oom else "failed"
                    res.failed_stage = "preprocessing"
                    res.elapsed_s    = time.perf_counter() - t_run
                    results.append(res)
                    continue

            completed_preproc.add(run.preproc_key)
        else:
            logger.info(
                "[%s] Preprocessing key '%s' already cached — skipping stages 2–4.",
                run.run_id, run.preproc_key,
            )

        # ------------------------------------------------------------------
        # Stage 5 : tiling
        # ------------------------------------------------------------------
        if run.tiling_key in failed_tiling:
            logger.warning(
                "[%s] Skipped — tiling key '%s' failed in an earlier run.",
                run.run_id, run.tiling_key,
            )
            res.status       = "failed"
            res.failed_stage = "tiling (inherited)"
            res.elapsed_s    = time.perf_counter() - t_run
            results.append(res)
            continue

        if run.tiling_key not in completed_tiling:
            tiling_cfg  = _patch_preproc_stage_5(base_preproc_cfg, run)
            tiling_yaml = _tiling_config_path(run.tiling_key)
            _write_yaml(tiling_cfg, tiling_yaml)
            logger.info("[%s] Tiling config → %s", run.run_id, tiling_yaml)

            if not dry_run:
                rc, oom = _run_stage(
                    cmd=[_PYTHON, "scripts/preprocessing.py",
                         "--config", str(tiling_yaml)],
                    run_id=run.run_id,
                    stage_label="tiling (stage 5)",
                    log_path=_stage_log_path(run.run_id, "tiling"),
                )
                if rc != 0:
                    failed_tiling.add(run.tiling_key)
                    res.status       = "oom" if oom else "failed"
                    res.failed_stage = "tiling"
                    res.elapsed_s    = time.perf_counter() - t_run
                    results.append(res)
                    continue

            completed_tiling.add(run.tiling_key)
        else:
            logger.info(
                "[%s] Tiling key '%s' already cached — skipping stage 5.",
                run.run_id, run.tiling_key,
            )

        # ------------------------------------------------------------------
        # Dataset YAML  (written every time — cheap, idempotent)
        # ------------------------------------------------------------------
        write_dataset_yaml(run)

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        train_cfg  = _patch_train_config(base_train_cfg, run)
        train_yaml = _train_config_path(run.run_id)
        _write_yaml(train_cfg, train_yaml)
        logger.info("[%s] Training config → %s", run.run_id, train_yaml)

        if dry_run:
            res.status = "skipped"
        else:
            rc, oom = _run_stage(
                cmd=[_PYTHON, "scripts/train.py", "--config", str(train_yaml)],
                run_id=run.run_id,
                stage_label="training",
                log_path=_stage_log_path(run.run_id, "training"),
            )
            if rc != 0:
                res.status       = "oom" if oom else "failed"
                res.failed_stage = "training"
                res.elapsed_s    = time.perf_counter() - t_run
                results.append(res)
                continue
            res.status = "success"

        res.elapsed_s = time.perf_counter() - t_run
        status_icon   = "○" if dry_run else "✓"
        logger.info(
            "[%s] %s complete in %.1fs.", run.run_id, status_icon, res.elapsed_s
        )
        results.append(res)

    return results


# =============================================================================
# Single-stage execution  (SLURM job-per-run mode)
# =============================================================================
# These functions are the entry-points when grid_search.py is invoked with
# --stage by submit_grid.py.  Each SLURM job calls exactly one of them for
# exactly one cache key / run-id.
#
# All YAML configs are generated up-front by submit_grid.py before any job
# starts, so these functions just assert the config exists and delegate to
# the appropriate pipeline script.  No patching or grid enumeration happens
# here — the config on disk is already the ground truth.
# =============================================================================

def _find_run_by_id(run_id: str) -> GridRun:
    """Return the :class:`GridRun` whose ``run_id`` matches *run_id*.

    Args:
        run_id: Unique run identifier, e.g. ``"up2_t1024_cl6_p2"``.

    Returns:
        Matching :class:`GridRun`.

    Raises:
        ValueError: If no run matches.
    """
    for run in build_grid():
        if run.run_id == run_id:
            return run
    raise ValueError(
        f"No GridRun found with run_id='{run_id}'.  "
        f"Valid IDs: {[r.run_id for r in build_grid()]}"
    )


def _find_run_by_tiling_key(tiling_key: str) -> GridRun:
    """Return the first :class:`GridRun` whose ``tiling_key`` matches.

    All runs sharing a tiling key produce an identical tiling config, so
    any sentinel is sufficient.

    Args:
        tiling_key: Tiling cache key, e.g. ``"up2_t1024_cl6"``.

    Returns:
        Any matching :class:`GridRun`.

    Raises:
        ValueError: If no run matches.
    """
    for run in build_grid():
        if run.tiling_key == tiling_key:
            return run
    raise ValueError(f"No GridRun found with tiling_key='{tiling_key}'.")


def _find_run_by_preproc_key(preproc_key: str) -> GridRun:
    """Return the first :class:`GridRun` whose ``preproc_key`` matches."""
    for run in build_grid():
        if run.preproc_key == preproc_key:
            return run
    raise ValueError(f"No GridRun found with preproc_key='{preproc_key}'.")


def execute_preproc_stage(key: str) -> int:
    """Run stages 2–4 for *key* as a standalone SLURM job.

    The YAML config is expected to already exist on disk (written by
    ``submit_grid.py`` during the config-generation pass).

    Args:
        key: Preprocessing cache key, e.g. ``"up2_cl6"``.

    Returns:
        Exit code of ``scripts/preprocessing.py``.
    """
    config_path = _preproc_config_path(key)
    if not config_path.exists():
        logger.critical(
            "Preproc config not found: %s  "
            "(run submit_grid.py --dry-run to regenerate configs)", config_path,
        )
        return 1

    logger.info("Executing preproc (stages 2–4) for key '%s'", key)
    rc, _ = _run_stage(
        cmd=[_PYTHON, "scripts/preprocessing.py", "--config", str(config_path)],
        run_id=key,
        stage_label="preproc",
        log_path=_stage_log_path(key, "preproc"),
    )
    return rc


def execute_tiling_stage(key: str) -> int:
    """Run stage 5 (tiling) for *key* as a standalone SLURM job.

    Also writes the ``dataset.yaml`` consumed by downstream training jobs.

    Args:
        key: Tiling cache key, e.g. ``"up2_t1024_cl6"``.

    Returns:
        Exit code of ``scripts/preprocessing.py``.
    """
    config_path = _tiling_config_path(key)
    if not config_path.exists():
        logger.critical(
            "Tiling config not found: %s  "
            "(run submit_grid.py --dry-run to regenerate configs)", config_path,
        )
        return 1

    # dataset.yaml is cheap and idempotent; write it here so the training job
    # can rely on it even if the file was lost or never written.
    sentinel = _find_run_by_tiling_key(key)
    write_dataset_yaml(sentinel)

    logger.info("Executing tiling (stage 5) for key '%s'", key)
    rc, _ = _run_stage(
        cmd=[_PYTHON, "scripts/preprocessing.py", "--config", str(config_path)],
        run_id=key,
        stage_label="tiling",
        log_path=_stage_log_path(key, "tiling"),
    )
    return rc


def execute_train_stage(run_id: str) -> int:
    """Run training for *run_id* as a standalone SLURM job.

    Args:
        run_id: Unique run identifier, e.g. ``"up2_t1024_cl6_p2"``.

    Returns:
        Exit code of ``scripts/train.py``.
    """
    config_path = _train_config_path(run_id)
    if not config_path.exists():
        logger.critical(
            "Training config not found: %s  "
            "(run submit_grid.py --dry-run to regenerate configs)", config_path,
        )
        return 1

    logger.info("Executing training for run '%s'", run_id)
    rc, oom = _run_stage(
        cmd=[_PYTHON, "scripts/train.py", "--config", str(config_path)],
        run_id=run_id,
        stage_label="training",
        log_path=_stage_log_path(run_id, "training"),
    )
    if oom:
        # Non-zero exit code is sufficient for SLURM to mark the job as failed
        # and prevent downstream jobs from starting via --kill-on-invalid-dep.
        logger.error("[%s] CUDA OOM — downstream jobs will be cancelled.", run_id)
    return rc


# =============================================================================
# Utilities
# =============================================================================

def _write_yaml(data: dict, path: Path) -> None:
    """Atomically write *data* as YAML to *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.yaml")
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    tmp.replace(path)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a plain dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed contents; empty dict if the file is blank.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _print_summary(results: List[RunResult]) -> None:
    """Log a results table and persist it to ``logs/grid_search_summary.txt``."""
    success = [r for r in results if r.status == "success"]
    oom     = [r for r in results if r.status == "oom"]
    failed  = [r for r in results if r.status not in ("success", "oom", "skipped")]
    skipped = [r for r in results if r.status == "skipped"]

    lines = [
        "",
        "=" * 72,
        f"GRID SEARCH SUMMARY  —  {len(results)} run(s)",
        "=" * 72,
        f"  ✓ Success  : {len(success)}",
        f"  ✗ OOM      : {len(oom)}",
        f"  ✗ Failed   : {len(failed)}",
        f"  ○ Skipped  : {len(skipped)}",
        "",
        f"{'RUN ID':<38} {'STATUS':<22} {'ELAPSED':>8}",
        "-" * 72,
    ]
    for r in results:
        note = f" [{r.failed_stage}]" if r.failed_stage else ""
        lines.append(
            f"{r.run_id:<38} {(r.status + note):<22} {r.elapsed_s:>7.0f}s"
        )
    lines.append("=" * 72)

    for line in lines:
        logger.info(line)

    summary_path = _ROOT / "logs" / "grid_search_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Summary → %s", summary_path)


# =============================================================================
# Logging
# =============================================================================

def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated grid search — maritime vessel detection pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--preproc-config",
        type=Path,
        default=Path("configs/preprocessing.yaml"),
        metavar="PATH",
        help="Base preprocessing config (default: configs/preprocessing.yaml).",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/train.yaml"),
        metavar="PATH",
        help="Base training config (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate all YAML configs and print the execution plan "
            "without running any preprocessing or training."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO).",
    )

    # ── Single-stage SLURM job mode ───────────────────────────────────────
    # These three arguments are set by submit_grid.py when submitting
    # individual SLURM jobs.  When --stage is present, only that one stage
    # is executed and the full grid loop is skipped entirely.
    slurm = parser.add_argument_group(
        "SLURM single-stage mode",
        "Set by submit_grid.py — execute exactly one pipeline stage.",
    )
    slurm.add_argument(
        "--stage",
        choices=["preproc", "tiling", "train"],
        default=None,
        metavar="STAGE",
        help="Stage to execute: preproc | tiling | train.",
    )
    slurm.add_argument(
        "--key",
        default=None,
        metavar="KEY",
        help="Cache key for --stage preproc or --stage tiling (e.g. up2_cl6).",
    )
    slurm.add_argument(
        "--run-id",
        default=None,
        metavar="RUN_ID",
        help="Run identifier for --stage train (e.g. up2_t1024_cl6_p2).",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, enumerate the grid, and orchestrate all runs.

    When ``--stage`` is provided (set by ``submit_grid.py``), executes a
    single pipeline stage for the given key / run-id and exits immediately.
    Otherwise runs the full sequential grid search.

    Args:
        argv: Optional argument list (``None`` → read ``sys.argv``).

    Returns:
        Exit code: ``0`` on full success, ``1`` if any run failed.
    """
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    # ── Single-stage SLURM dispatch ───────────────────────────────────────
    if args.stage is not None:
        if args.stage in ("preproc", "tiling") and args.key is None:
            logger.critical("--stage %s requires --key.", args.stage)
            return 1
        if args.stage == "train" and args.run_id is None:
            logger.critical("--stage train requires --run-id.")
            return 1

        if args.stage == "preproc":
            return execute_preproc_stage(args.key)
        if args.stage == "tiling":
            return execute_tiling_stage(args.key)
        # args.stage == "train"
        return execute_train_stage(args.run_id)

    # ── Full sequential grid search (monolithic mode) ─────────────────────
    logger.info("Vessel Detection — YOLO-OBB Grid Search")
    logger.info("  Upscale ratios : %s", UPSCALE_RATIOS)
    logger.info("  Tile sizes     : %s", TILE_SIZES)
    logger.info("  Class configs  : %s", sorted(CLASS_CONFIGS.keys()))
    logger.info("  Architectures  : %s", sorted(ARCH_CONFIGS.keys()))

    base_preproc_cfg = _load_yaml(args.preproc_config.resolve())
    base_train_cfg   = _load_yaml(args.train_config.resolve())

    runs = build_grid()
    logger.info("")
    logger.info("Grid: %d runs total", len(runs))
    for i, r in enumerate(runs, 1):
        logger.info("  %02d. %s", i, r.run_id)

    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN — configs will be written but nothing will execute.")

    t_total = time.perf_counter()
    results = run_grid_search(
        runs=runs,
        base_preproc_cfg=base_preproc_cfg,
        base_train_cfg=base_train_cfg,
        dry_run=args.dry_run,
    )
    total_elapsed = time.perf_counter() - t_total

    logger.info("")
    logger.info(
        "Total elapsed: %.1f s  (%.1f h)", total_elapsed, total_elapsed / 3600
    )
    _print_summary(results)

    n_hard_failures = sum(
        1 for r in results if r.status in ("failed", "oom")
    )
    return 1 if (n_hard_failures > 0 and not args.dry_run) else 0


if __name__ == "__main__":
    sys.exit(main())