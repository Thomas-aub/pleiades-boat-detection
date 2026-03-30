"""
src/vessels_detect/postprocessing/manager.py
---------------------------------------------
Registry/Manager for the GeoJSON Postprocessing Pipeline.

Architecture overview
~~~~~~~~~~~~~~~~~~~~~
::

    manager.py  (this file)
    │
    ├── STEP_REGISTRY  — maps stage name → BaseStep subclass
    │
    ├── load_config()  — parse YAML + resolve paths
    │
    └── PostprocessingManager
            run()  →  iterates over enabled stages in ``stages`` list
                       instantiates the registered step, calls step.run(cfg)

The postprocessing pipeline operates entirely on GeoJSON files and has
**no dependency** on model weights, inference code, or rasterio tiles.
Its only inputs are:

- ``predicted/``    — raw model output GeoJSON files.
- ``coastlines/``   — per-image coastline boundary GeoJSON masks.
- ``buildings/``    — per-image building footprint GeoJSON masks.
- ``postprocessed/``— output directory (created automatically).

Adding a new filter step
~~~~~~~~~~~~~~~~~~~~~~~~
1.  Create a module in ``src/vessels_detect/postprocessing/steps/``.
2.  Subclass :class:`~steps.base.BaseStep` and set a unique ``NAME``.
3.  Register it in :data:`STEP_REGISTRY` below — zero changes elsewhere.
4.  Add the corresponding section to ``configs/postprocessing.yaml``.

Typical usage::

    from src.vessels_detect.postprocessing.manager import PostprocessingManager

    manager = PostprocessingManager(config_path=Path("configs/postprocessing.yaml"))
    manager.run()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

import yaml

from src.vessels_detect.postprocessing.steps.base      import BaseStep
from src.vessels_detect.postprocessing.steps.coastline import CoastlineFilter
from src.vessels_detect.postprocessing.steps.buildings import BuildingsFilter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

#: Maps step name (``BaseStep.NAME``) → step class.
#: Add new steps here — no other file needs to change.
STEP_REGISTRY: Dict[str, Type[BaseStep]] = {
    CoastlineFilter.NAME: CoastlineFilter,
    BuildingsFilter.NAME: BuildingsFilter,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load, validate, and resolve the postprocessing YAML configuration.

    Path strings in ``cfg["paths"]`` are converted to absolute
    :class:`~pathlib.Path` objects resolved relative to the current working
    directory (expected to be the project root).

    Args:
        config_path: Path to ``configs/postprocessing.yaml``.

    Returns:
        Fully resolved configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        yaml.YAMLError:    If the file is not valid YAML.
        KeyError:          If a required top-level key is missing.
        ValueError:        If ``stages`` references an unknown step name.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)

    for section in ("paths", "stages"):
        if section not in cfg:
            raise KeyError(
                f"Required section '{section}' missing from '{config_path}'."
            )

    cfg["paths"] = {k: Path(v).resolve() for k, v in cfg["paths"].items()}

    for stage_entry in cfg["stages"]:
        name = stage_entry.get("name", "")
        if name not in STEP_REGISTRY:
            raise ValueError(
                f"Stage '{name}' is not registered.  "
                f"Known stages: {sorted(STEP_REGISTRY)}."
            )

    return cfg


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class PostprocessingManager:
    """Orchestrates the postprocessing pipeline based on the YAML config.

    The manager reads the ``stages`` list from the configuration and
    executes only the steps that are both listed and have ``enabled: true``
    (default when the key is absent).

    Args:
        config_path: Path to ``configs/postprocessing.yaml``.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._cfg: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, stages_override: Optional[List[str]] = None) -> int:
        """Load config and execute the requested pipeline stages.

        Stages run in the order defined in the YAML.  When
        :class:`~steps.coastline.CoastlineFilter` precedes
        :class:`~steps.buildings.BuildingsFilter`, the buildings filter
        automatically reads from ``postprocessed/`` (controlled by
        ``read_from_postprocessed`` in the YAML).

        Args:
            stages_override: If provided, only these stage names are run
                (``enabled`` flags are ignored).

        Returns:
            Exit code: ``0`` on success, ``1`` on fatal error.
        """
        try:
            self._cfg = load_config(self._config_path)
        except Exception as exc:  # noqa: BLE001
            logger.critical("Failed to load config: %s", exc)
            return 1

        cfg = self._cfg

        logger.info("GeoJSON Postprocessing Pipeline")
        logger.info("Config : %s", self._config_path.resolve())
        logger.info("Paths:")
        for key, val in cfg["paths"].items():
            logger.info("  %-20s: %s", key, val)

        stages_to_run = self._resolve_stages(cfg["stages"], stages_override)

        if not stages_to_run:
            logger.warning("No stages are enabled.  Nothing to do.")
            return 0

        logger.info("Stages to run: %s", [s["name"] for s in stages_to_run])

        pipeline_start = time.perf_counter()
        total_removed  = 0

        for stage_entry in stages_to_run:
            name     = stage_entry["name"]
            stage_id = stage_entry.get("id", "?")
            step_cls = STEP_REGISTRY[name]
            step     = step_cls()

            logger.info("")
            logger.info("=" * 70)
            logger.info("STAGE %s — %s", stage_id, name.upper())
            logger.info("=" * 70)

            t0 = time.perf_counter()
            try:
                removed = step.run(cfg)
            except Exception as exc:  # noqa: BLE001
                logger.critical(
                    "Stage %s (%s) failed: %s", stage_id, name, exc,
                    exc_info=True,
                )
                return 1

            total_removed += removed
            elapsed = time.perf_counter() - t0
            logger.info(
                "Stage %s (%s) — %d polygon(s) removed in %.1f s.",
                stage_id, name, removed, elapsed,
            )

        total = time.perf_counter() - pipeline_start
        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "Pipeline complete.  Total polygons removed: %d  |  Time: %.1f s",
            total_removed, total,
        )
        logger.info("=" * 70)
        return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_stages(
        stage_list: List[dict],
        override: Optional[List[str]],
    ) -> List[dict]:
        """Filter the stage list to the steps that should be executed.

        Args:
            stage_list: ``cfg["stages"]`` list from the YAML config.
            override:   Optional explicit list of stage names; preserves
                        YAML order.

        Returns:
            Ordered list of stage entry dicts to execute.
        """
        if override is not None:
            override_set = set(override)
            return [s for s in stage_list if s.get("name") in override_set]
        return [s for s in stage_list if s.get("enabled", True)]