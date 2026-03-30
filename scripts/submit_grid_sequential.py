#!/usr/bin/env python3
"""
scripts/submit_grid_sequential.py
----------------------------------
"Good citizen" sequential SLURM manager for the maritime grid search.

Instead of flooding the queue with all 60 jobs at once, this script acts as
a lightweight self-rescheduling daemon: it submits exactly ONE real job, then
re-queues itself as a dependency of that job.  When the real job finishes
(success OR failure), the manager wakes up, advances a persistent cursor by
one step, submits the next real job, and goes back to sleep.

At any given moment the queue contains at most TWO entries:
    1. The active GPU/CPU job doing real work.
    2. The sleeping manager job waiting to dispatch the next step.

This means other users can always slip jobs between our runs.  The manager
itself runs on a minimal CPU allocation (1 core, 1 GB RAM, 2-minute walltime)
so it wastes essentially no cluster resources.

Queue snapshot while running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    JOBID   NAME          STATE   REASON
    11001   GS_mgr_042    PEND    Dependency      ← manager sleeping
    11000   GS_train_...  RUN     None            ← real work happening

After the GPU job finishes, 11001 starts, submits step 043, and exits in
under five seconds.

Execution plan
~~~~~~~~~~~~~~
The 60 steps are linearised in dependency order:

    step 000-005  : preproc  (6 unique keys,  CPU)
    step 006-023  : tiling   (18 unique keys, CPU)
    step 024-059  : training (36 runs,        GPU)

Within each group the ordering is deterministic (same as build_grid()).
Preproc steps for a given upscale/class key are guaranteed to precede all
tiling steps that depend on them, which precede all training steps.

State persistence
~~~~~~~~~~~~~~~~~
``logs/grid_search_seq_state.json`` tracks the cursor (next step index) and
the last submitted job ID.  The file is written atomically before every real
job submission so a crash of the manager is always recoverable by re-running
this script — it will simply pick up where it left off.

Usage
~~~~~
::

    # Preview the linearised execution plan without submitting anything
    PYTHONPATH=. python scripts/submit_grid_sequential.py --dry-run

    # Kick off the sequential grid search (submits step 0 + manager job 1)
    PYTHONPATH=. python scripts/submit_grid_sequential.py

    # Specify partitions explicitly
    PYTHONPATH=. python scripts/submit_grid_sequential.py \\
        --cpu-partition cpu \\
        --gpu-partition gpu

    # The manager re-invokes itself automatically; you never need to call
    # this script again unless you want to inspect state or restart.

    # Check progress at any time
    cat logs/grid_search_seq_state.json
    squeue -u $USER

    # Cancel everything cleanly
    scancel --name=GS_mgr   # stops the manager chain
    # (any already-running real job will complete normally)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Bootstrap: import grid_search from the same scripts/ directory.
# =============================================================================

_THIS_DIR  = Path(__file__).resolve().parent
_ROOT      = _THIS_DIR.parent
_GS_MODULE = _THIS_DIR / "grid_search.py"

_spec = importlib.util.spec_from_file_location("grid_search", _GS_MODULE)
_gs   = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
sys.modules["grid_search"] = _gs
_spec.loader.exec_module(_gs)                            # type: ignore[union-attr]

GridRun       = _gs.GridRun
build_grid    = _gs.build_grid
_write_yaml   = _gs._write_yaml
_load_yaml    = _gs._load_yaml

_patch_preproc_stages_2_to_4 = _gs._patch_preproc_stages_2_to_4
_patch_preproc_stage_5       = _gs._patch_preproc_stage_5
_patch_train_config          = _gs._patch_train_config
write_dataset_yaml            = _gs.write_dataset_yaml

_preproc_config_path = _gs._preproc_config_path
_tiling_config_path  = _gs._tiling_config_path
_train_config_path   = _gs._train_config_path

_PYTHON       = sys.executable
_GS_LOGS_ROOT = _ROOT / "logs" / "grid_search"
_STATE_PATH   = _ROOT / "logs" / "grid_search_seq_state.json"

# Directory where ephemeral sbatch bash wrappers are written.
# Each job gets its own file so concurrent dry-run inspection is safe.
_SBATCH_SCRIPTS_DIR = _ROOT / "logs" / "grid_search" / "sbatch_scripts"

logger = logging.getLogger(__name__)


# =============================================================================
# SLURM resource profiles  (unchanged from submit_grid.py)
# =============================================================================

@dataclass(frozen=True)
class SlurmProfile:
    """SLURM resource specification for one job type."""

    job_prefix: str
    partition:  str
    cpus:       int
    mem_gb:     int
    walltime:   str
    gres:       str = ""


_TRAIN_WALLTIME: Dict[int, str] = {
    640:  "04:00:00",
    1024: "08:00:00",
    1536: "12:00:00",
}

# Manager profile: tiny footprint — it only calls sbatch and exits.
_MANAGER_PROFILE = SlurmProfile(
    job_prefix="GS_mgr",
    partition="",          # filled in at runtime from --cpu-partition
    cpus=1,
    mem_gb=1,
    walltime="00:05:00",   # 5 minutes is more than enough to call sbatch
)


def _make_profiles(
    gpu_partition: str,
    cpu_partition: str,
    train_walltime_override: Optional[str],
) -> Tuple[SlurmProfile, SlurmProfile, Dict[int, SlurmProfile], SlurmProfile]:
    """Build resource profiles for all stage types plus the manager."""
    preproc_profile = SlurmProfile(
        job_prefix="GS_prep",
        partition=cpu_partition,
        cpus=4,
        mem_gb=128,
        walltime="02:00:00",
    )
    tiling_profile = SlurmProfile(
        job_prefix="GS_tile",
        partition=cpu_partition,
        cpus=4,
        mem_gb=32,
        walltime="01:00:00",
    )
    train_profiles = {
        tile_size: SlurmProfile(
            job_prefix="GS_train",
            partition=gpu_partition,
            cpus=8,
            mem_gb=64,
            walltime=train_walltime_override or wt,
            gres="gpu:1",
        )
        for tile_size, wt in _TRAIN_WALLTIME.items()
    }
    manager_profile = SlurmProfile(
        job_prefix="GS_mgr",
        partition=cpu_partition,
        cpus=1,
        mem_gb=1,
        walltime="00:05:00",
    )
    return preproc_profile, tiling_profile, train_profiles, manager_profile


# =============================================================================
# Step descriptor — the linearised execution plan
# =============================================================================

@dataclass(frozen=True)
class Step:
    """One entry in the sequential execution plan."""

    index:     int
    stage:     str    # "preproc" | "tiling" | "train"
    key:       str    # preproc/tiling cache key, or run_id for training
    job_name:  str
    log_stem:  str


def build_plan(runs: List[GridRun]) -> List[Step]:
    """Linearise the 60-job DAG into a sequential list of Steps.

    Ordering guarantees:
        • All preproc steps precede all tiling steps.
        • Each tiling step follows its upstream preproc step.
        • Each training step follows its upstream tiling step.

    Within each stage group the order is deterministic (mirrors build_grid).

    Args:
        runs: All 36 GridRun descriptors from build_grid().

    Returns:
        Ordered list of 60 Steps.
    """
    steps: List[Step] = []

    # --- Stage 2-4: preproc (6 unique keys) --------------------------------
    seen_preproc: dict = {}
    for run in runs:
        if run.preproc_key not in seen_preproc:
            seen_preproc[run.preproc_key] = True
            idx = len(steps)
            steps.append(Step(
                index=idx,
                stage="preproc",
                key=run.preproc_key,
                job_name=f"GS_prep_{run.preproc_key}",
                log_stem=f"preproc_{run.preproc_key}",
            ))

    # --- Stage 5: tiling (18 unique keys) ----------------------------------
    seen_tiling: dict = {}
    for run in runs:
        if run.tiling_key not in seen_tiling:
            seen_tiling[run.tiling_key] = True
            idx = len(steps)
            steps.append(Step(
                index=idx,
                stage="tiling",
                key=run.tiling_key,
                job_name=f"GS_tile_{run.tiling_key}",
                log_stem=f"tiling_{run.tiling_key}",
            ))

    # --- Training (36 runs) ------------------------------------------------
    for run in runs:
        idx = len(steps)
        steps.append(Step(
            index=idx,
            stage="train",
            key=run.run_id,
            job_name=f"GS_train_{run.run_id}",
            log_stem=f"train_{run.run_id}",
        ))

    return steps


# =============================================================================
# State persistence
# =============================================================================

def _load_state() -> dict:
    """Load the sequential cursor state from disk.

    Returns a dict with keys:
        ``next_step``     : int  — index of the step to submit next
        ``last_job_id``   : int  — SLURM job ID of the most recently submitted
                                   real job (0 if none yet)
        ``total_steps``   : int  — total number of steps in the plan (60)
    """
    if _STATE_PATH.exists():
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    return {"next_step": 0, "last_job_id": 0, "total_steps": 0}


def _save_state(state: dict) -> None:
    """Atomically persist *state* to ``_STATE_PATH``."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_PATH.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(_STATE_PATH)
    logger.debug("State saved → %s", _STATE_PATH)


# =============================================================================
# sbatch helpers
# =============================================================================

def _write_bash_script(script_name: str, cmd_args: List[str]) -> Path:
    """Write a minimal bash wrapper script for sbatch and return its path.

    Using a real bash file (instead of ``--wrap``) is required on clusters
    that restrict inline job submission, and also produces cleaner audit
    trails.  Each script is written to ``_SBATCH_SCRIPTS_DIR`` and kept
    after the run for post-mortem inspection.

    Args:
        script_name: Filename stem (no extension), e.g. ``"preproc_up2_cl6"``.
        cmd_args:    Full command to execute, e.g.
                     ``["/path/to/python3.12", "scripts/grid_search.py", ...]``.

    Returns:
        Absolute path to the written ``.sh`` file.
    """
    _SBATCH_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    bash_path = _SBATCH_SCRIPTS_DIR / f"{script_name}.sh"

    cmd_str = " ".join(f'"{a}"' if " " in str(a) else str(a) for a in cmd_args)

    content = (
        "#!/bin/bash\n"
        f"# Auto-generated by submit_grid_sequential.py\n"
        f"cd {_ROOT}\n"
        f"export PYTHONPATH={_ROOT}\n"
        f"{cmd_str}\n"
    )
    bash_path.write_text(content, encoding="utf-8")
    bash_path.chmod(0o755)
    return bash_path


def _sbatch_submit(
    sbatch_args: List[str],
    bash_path: Path,
    dry_run: bool,
) -> int:
    """Submit one SLURM job via a bash script file and return its job ID.

    Args:
        sbatch_args: ``sbatch`` flags (without the script path).
        bash_path:   Path to the bash wrapper script to submit.
        dry_run:     Log without submitting; return a synthetic ID.

    Returns:
        SLURM job ID (real or synthetic).

    Raises:
        subprocess.CalledProcessError: If sbatch exits non-zero.
    """
    cmd = ["sbatch", "--parsable"] + sbatch_args + [str(bash_path)]

    if dry_run:
        _sbatch_submit._counter += 1          # type: ignore[attr-defined]
        fake_id = _sbatch_submit._counter
        logger.info("  [DRY-RUN] %s  →  job_id=%d", " ".join(cmd), fake_id)
        return fake_id

    logger.debug("  sbatch: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=str(_ROOT),
    )
    raw    = result.stdout.strip().split(";")[0]
    job_id = int(raw)
    logger.debug("  → job_id=%d", job_id)
    return job_id


_sbatch_submit._counter = 0  # type: ignore[attr-defined]


def _build_sbatch_args(
    profile: SlurmProfile,
    job_name: str,
    log_stem: str,
    dependency_job_id: Optional[int] = None,
    dependency_type: str = "afterany",
) -> List[str]:
    """Build sbatch flag list for one job.

    Args:
        profile:           Resource profile.
        job_name:          Value for ``--job-name``.
        log_stem:          Stem for stdout/stderr log filenames.
        dependency_job_id: If set, add ``--dependency=<type>:<id>``.
        dependency_type:   ``"afterok"`` (real jobs) or ``"afterany"``
                           (manager — must wake up even on failure).

    Returns:
        List of sbatch flag strings.
    """
    _GS_LOGS_ROOT.mkdir(parents=True, exist_ok=True)

    args = [
        f"--job-name={job_name}",
        f"--partition={profile.partition}",
        "-c", str(profile.cpus),
        f"--mem={profile.mem_gb}G",
        f"--time={profile.walltime}",
        "--nodes=1",
        "-o", str(_GS_LOGS_ROOT / f"{log_stem}_%j.out"),
        "-e", str(_GS_LOGS_ROOT / f"{log_stem}_%j.err"),
    ]

    if profile.gres:
        args += ["--gres", profile.gres]

    if dependency_job_id is not None:
        args += [f"--dependency={dependency_type}:{dependency_job_id}"]

    return args


# =============================================================================
# Config generation (all YAMLs written once at bootstrap)
# =============================================================================

def generate_all_configs(
    runs: List[GridRun],
    base_preproc_cfg: dict,
    base_train_cfg: dict,
) -> None:
    """Write every YAML config to disk before any job is submitted.

    This mirrors the Pass-1 logic from submit_grid.py so that each SLURM
    job (including those spawned days later) always finds its config file.

    Args:
        runs:             All 36 GridRun descriptors.
        base_preproc_cfg: Parsed ``configs/preprocessing.yaml``.
        base_train_cfg:   Parsed ``configs/train.yaml``.
    """
    logger.info("Generating YAML configs for all 60 jobs …")

    preproc_seen: dict = {}
    tiling_seen:  dict = {}

    for run in runs:
        if run.preproc_key not in preproc_seen:
            preproc_seen[run.preproc_key] = run
            cfg = _patch_preproc_stages_2_to_4(base_preproc_cfg, run)
            _write_yaml(cfg, _preproc_config_path(run.preproc_key))

        if run.tiling_key not in tiling_seen:
            tiling_seen[run.tiling_key] = run
            cfg = _patch_preproc_stage_5(base_preproc_cfg, run)
            _write_yaml(cfg, _tiling_config_path(run.tiling_key))
            write_dataset_yaml(run)

        cfg = _patch_train_config(base_train_cfg, run)
        _write_yaml(cfg, _train_config_path(run.run_id))

    logger.info(
        "  %d preproc / %d tiling / %d training configs written.",
        len(preproc_seen), len(tiling_seen), len(runs),
    )


# =============================================================================
# Core dispatch logic
# =============================================================================

def _step_profile(
    step: Step,
    train_profiles: Dict[int, SlurmProfile],
    preproc_profile: SlurmProfile,
    tiling_profile: SlurmProfile,
) -> SlurmProfile:
    """Return the resource profile for *step*."""
    if step.stage == "preproc":
        return preproc_profile
    if step.stage == "tiling":
        return tiling_profile
    # Training: look up by tile size embedded in the run_id (e.g. "up2_t1024_cl6_p2")
    for tile_size in sorted(train_profiles.keys(), reverse=True):
        if f"_t{tile_size}_" in step.key:
            return train_profiles[tile_size]
    # Fallback to 640 if parsing fails (should not happen with valid run IDs)
    logger.warning("Could not infer tile size from run_id '%s'; using 640 profile.", step.key)
    return train_profiles[640]


def _step_cmd(step: Step) -> List[str]:
    """Return the full command list for a real work step.

    Uses absolute paths for both the interpreter and the script so the
    generated bash file is valid regardless of working directory at
    submission time.
    """
    gs_script = str(_THIS_DIR / "grid_search.py")
    if step.stage == "train":
        return [_PYTHON, gs_script, "--stage", "train", "--run-id", step.key]
    return [_PYTHON, gs_script, "--stage", step.stage, "--key", step.key]


def submit_next_step(
    plan: List[Step],
    state: dict,
    preproc_profile: SlurmProfile,
    tiling_profile: SlurmProfile,
    train_profiles: Dict[int, SlurmProfile],
    manager_profile: SlurmProfile,
    gpu_partition: str,
    dry_run: bool,
) -> bool:
    """Submit the next real job and re-queue the manager behind it.

    Both the real job and the manager are submitted via bash script files
    (not ``--wrap``) so the submission works on clusters that require it
    and produces clean audit trails under ``logs/grid_search/sbatch_scripts/``.

    Flow:
        1. Write a bash script for the real work step and submit it.
        2. Persist the cursor to disk (crash-safe).
        3. Write a bash script that re-runs *this manager* and submit it
           with ``--dependency=afterany:<real_job_id>`` so it wakes up
           when the real job ends, regardless of its exit status.

    Args:
        plan:            Ordered list of all 60 Steps.
        state:           Mutable state dict (updated in place).
        preproc_profile: Resource profile for preproc jobs.
        tiling_profile:  Resource profile for tiling jobs.
        train_profiles:  Resource profiles by tile size.
        manager_profile: Resource profile for the manager itself.
        gpu_partition:   Name of the GPU partition (forwarded to manager re-invocation).
        dry_run:         Log without submitting.

    Returns:
        ``True`` if a step was submitted, ``False`` if all steps are done.
    """
    next_idx = state["next_step"]

    if next_idx >= len(plan):
        logger.info("All %d steps completed.  Grid search is done.", len(plan))
        return False

    step    = plan[next_idx]
    profile = _step_profile(step, train_profiles, preproc_profile, tiling_profile)

    logger.info(
        "Submitting step %d / %d  [%s]  %s",
        next_idx + 1, len(plan), step.stage, step.key,
    )

    # -- 1. Write bash script + submit real work job ---------------------------
    real_bash = _write_bash_script(
        script_name=step.log_stem,
        cmd_args=_step_cmd(step),
    )
    real_sbatch_args = _build_sbatch_args(
        profile=profile,
        job_name=step.job_name,
        log_stem=step.log_stem,
    )
    real_job_id = _sbatch_submit(
        sbatch_args=real_sbatch_args,
        bash_path=real_bash,
        dry_run=dry_run,
    )
    logger.info("  -> real job_id=%d  (%s)", real_job_id, real_bash.name)

    # -- 2. Persist cursor BEFORE submitting the manager (crash safety) --------
    state["next_step"]   = next_idx + 1
    state["last_job_id"] = real_job_id
    _save_state(state)

    # -- 3. Re-queue this manager script as a dependency of the real job -------
    #    afterany  ->  manager wakes up even if the real job fails, so the
    #                  grid search keeps advancing through subsequent steps.
    if next_idx + 1 < len(plan):
        mgr_step_label = next_idx + 1
        mgr_cmd = [
            _PYTHON,
            str(Path(__file__).resolve()),
            "--gpu-partition", gpu_partition,
            "--cpu-partition", manager_profile.partition,
        ]
        if dry_run:
            mgr_cmd += ["--dry-run"]

        mgr_bash = _write_bash_script(
            script_name=f"mgr_{mgr_step_label:03d}",
            cmd_args=mgr_cmd,
        )
        mgr_sbatch_args = _build_sbatch_args(
            profile=manager_profile,
            job_name=f"GS_mgr_{mgr_step_label:03d}",
            log_stem=f"mgr_{mgr_step_label:03d}",
            dependency_job_id=real_job_id,
            dependency_type="afterany",
        )
        mgr_job_id = _sbatch_submit(
            sbatch_args=mgr_sbatch_args,
            bash_path=mgr_bash,
            dry_run=dry_run,
        )
        logger.info(
            "  -> manager job_id=%d  (wakes after job %d)", mgr_job_id, real_job_id
        )
    else:
        logger.info("  Last step submitted -- no manager re-queue needed.")

    return True



# =============================================================================
# Bootstrap: first invocation (cursor == 0)
# =============================================================================

def bootstrap(
    runs: List[GridRun],
    plan: List[Step],
    base_preproc_cfg: dict,
    base_train_cfg: dict,
    preproc_profile: SlurmProfile,
    tiling_profile: SlurmProfile,
    train_profiles: Dict[int, SlurmProfile],
    manager_profile: SlurmProfile,
    gpu_partition: str,
    dry_run: bool,
) -> None:
    """First-time setup: generate configs, initialise state, kick off step 0.

    Args:
        runs:             All 36 GridRun descriptors.
        plan:             Linearised 60-step plan.
        base_preproc_cfg: Parsed ``configs/preprocessing.yaml``.
        base_train_cfg:   Parsed ``configs/train.yaml``.
        preproc_profile:  Resource profile for preproc.
        tiling_profile:   Resource profile for tiling.
        train_profiles:   Resource profiles by tile size.
        manager_profile:  Resource profile for manager.
        gpu_partition:    Name of the GPU partition.
        dry_run:          Log without submitting.
    """
    logger.info("Bootstrap: generating all YAML configs …")
    generate_all_configs(runs, base_preproc_cfg, base_train_cfg)

    state = {"next_step": 0, "last_job_id": 0, "total_steps": len(plan)}
    _save_state(state)

    submit_next_step(
        plan=plan,
        state=state,
        preproc_profile=preproc_profile,
        tiling_profile=tiling_profile,
        train_profiles=train_profiles,
        manager_profile=manager_profile,
        gpu_partition=gpu_partition,
        dry_run=dry_run,
    )


# =============================================================================
# Summary / plan display
# =============================================================================

def _print_plan(plan: List[Step]) -> None:
    """Log the full sequential execution plan."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("SEQUENTIAL GRID SEARCH PLAN  —  %d steps (1 job at a time)", len(plan))
    logger.info("=" * 72)
    for step in plan:
        logger.info("  %3d.  [%-7s]  %s", step.index + 1, step.stage, step.key)
    logger.info("=" * 72)


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


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Good-citizen sequential SLURM manager: "
            "submits one job at a time and re-queues itself as a dependency."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gpu-partition", default="gpu", metavar="NAME",
        help="SLURM partition for training jobs (default: gpu).",
    )
    parser.add_argument(
        "--cpu-partition", default=None, metavar="NAME",
        help=(
            "SLURM partition for preproc / tiling / manager jobs "
            "(default: same as --gpu-partition)."
        ),
    )
    parser.add_argument(
        "--train-walltime", default=None, metavar="HH:MM:SS",
        help=(
            "Override training walltime for all tile sizes "
            "(default: 04h/08h/12h for 640/1024/1536 px)."
        ),
    )
    parser.add_argument(
        "--preproc-config",
        type=Path, default=Path("configs/preprocessing.yaml"), metavar="PATH",
        help="Base preprocessing config (default: configs/preprocessing.yaml).",
    )
    parser.add_argument(
        "--train-config",
        type=Path, default=Path("configs/train.yaml"), metavar="PATH",
        help="Base training config (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the full plan and simulate submission without touching SLURM.",
    )
    parser.add_argument(
        "--print-plan", action="store_true",
        help="Print the linearised execution plan and exit (no submission).",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help=(
            "Delete the state file and restart from step 0.  "
            "Use after cancelling a run to start fresh."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], metavar="LEVEL",
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for both the first invocation and all manager wake-ups.

    On first call (cursor == 0 or ``--reset``): generates configs, writes
    state, and submits step 0 + manager 1.

    On subsequent calls (spawned by SLURM as a dependency job): reads the
    cursor, submits the next real job, and re-queues itself.

    Args:
        argv: Optional argument list (``None`` → read ``sys.argv``).

    Returns:
        Exit code: 0 on success, 1 on sbatch error.
    """
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    gpu_part = args.gpu_partition
    cpu_part = args.cpu_partition or gpu_part

    logger.info("Vessel Detection — Sequential Grid Search Manager")
    logger.info("  GPU partition : %s", gpu_part)
    logger.info("  CPU partition : %s", cpu_part)
    if args.dry_run:
        logger.info("  Mode          : DRY-RUN (no jobs submitted)")

    # ── Handle --reset ────────────────────────────────────────────────────────
    if args.reset:
        if _STATE_PATH.exists():
            _STATE_PATH.unlink()
            logger.info("State file removed: %s", _STATE_PATH)
        else:
            logger.info("No state file found — nothing to reset.")

    # ── Build plan ────────────────────────────────────────────────────────────
    runs = build_grid()
    plan = build_plan(runs)

    if args.print_plan:
        _print_plan(plan)
        return 0

    preproc_profile, tiling_profile, train_profiles, manager_profile = _make_profiles(
        gpu_partition=gpu_part,
        cpu_partition=cpu_part,
        train_walltime_override=args.train_walltime,
    )

    # ── Load state ────────────────────────────────────────────────────────────
    state = _load_state()
    next_step = state.get("next_step", 0)

    logger.info(
        "State: step %d / %d  (last real job: %d)",
        next_step, len(plan), state.get("last_job_id", 0),
    )

    try:
        if next_step == 0:
            # First-time bootstrap: generate configs + kick off step 0.
            base_preproc_cfg = _load_yaml(args.preproc_config.resolve())
            base_train_cfg   = _load_yaml(args.train_config.resolve())
            bootstrap(
                runs=runs,
                plan=plan,
                base_preproc_cfg=base_preproc_cfg,
                base_train_cfg=base_train_cfg,
                preproc_profile=preproc_profile,
                tiling_profile=tiling_profile,
                train_profiles=train_profiles,
                manager_profile=manager_profile,
                gpu_partition=gpu_part,
                dry_run=args.dry_run,
            )
        else:
            # Manager woke up after a real job finished — submit next step.
            done = not submit_next_step(
                plan=plan,
                state=state,
                preproc_profile=preproc_profile,
                tiling_profile=tiling_profile,
                train_profiles=train_profiles,
                manager_profile=manager_profile,
                gpu_partition=gpu_part,
                dry_run=args.dry_run,
            )
            if done:
                logger.info("Sequential grid search complete.")

    except subprocess.CalledProcessError as exc:
        logger.critical("sbatch failed: %s\n%s", exc, exc.stderr)
        return 1

    logger.info("Monitor:  squeue -u $USER")
    logger.info("State:    cat %s", _STATE_PATH)
    logger.info("Cancel:   scancel --name=GS_mgr && scancel --name=GS_prep "
                "&& scancel --name=GS_tile && scancel --name=GS_train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
