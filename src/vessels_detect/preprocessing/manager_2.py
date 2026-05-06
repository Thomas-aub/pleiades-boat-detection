"""
src/vessels_detect/preprocessing/manager_2.py
----------------------------------------------
Registry/Manager for the SAHI-optimised Global Preprocessing Pipeline v2.

Differences from ``manager.py`` (v1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Stages 4 (split) and 5 (tiling) are **replaced** by Stage 6 (tiled_split).
- The tiled_split step tiles all processed GeoTIFFs first (no overlap), then
  splits and shards the tile pool into the new output layout.
- A ``metadata.csv`` is written to the output root automatically by the step.

Step registry (v2)
~~~~~~~~~~~~~~~~~~
::

    radiometric  → RadiometricStep
    spatial      → SpatialStep
    annotations  → AnnotationStep
    tiled_split  → TiledSplitStep   ← new unified step

Adding a new step
~~~~~~~~~~~~~~~~~
1. Create a module in ``src/vessels_detect/preprocessing/steps/``.
2. Subclass :class:`~steps.base.BaseStep` and set a unique ``NAME``.
3. Register it in :data:`STEP_REGISTRY` below.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

import yaml

from src.vessels_detect.preprocessing.steps.base import BaseStep
from src.vessels_detect.preprocessing.steps.radiometric  import RadiometricStep
from src.vessels_detect.preprocessing.steps.spatial      import SpatialStep
from src.vessels_detect.preprocessing.steps.annotations  import AnnotationStep
from src.vessels_detect.preprocessing.steps.tiled_split  import TiledSplitStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step registry (v2 — no split / tiling; tiled_split replaces both)
# ---------------------------------------------------------------------------

STEP_REGISTRY: Dict[str, Type[BaseStep]] = {
    RadiometricStep.NAME:  RadiometricStep,
    SpatialStep.NAME:      SpatialStep,
    AnnotationStep.NAME:   AnnotationStep,
    TiledSplitStep.NAME:   TiledSplitStep,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load, validate, and resolve the preprocessing YAML configuration.

    Identical behaviour to ``manager.load_config`` but validates against the
    v2 step registry.

    Args:
        config_path: Path to ``configs/preprocessing_2.yaml``.

    Returns:
        Fully resolved configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        KeyError: If a required top-level key is missing.
        ValueError: If the ``stages`` list references an unknown step name.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as fh:
        cfg: dict = yaml.safe_load(fh)

    for section in ("paths", "stages"):
        if section not in cfg:
            raise KeyError(
                f"Required section '{section}' is missing from '{config_path}'."
            )

    # Resolve all path strings to absolute Path objects.
    cfg["paths"] = {k: Path(v).resolve() for k, v in cfg["paths"].items()}

    # Validate stage names against the v2 registry.
    for stage_entry in cfg["stages"]:
        name = stage_entry.get("name", "")
        if name not in STEP_REGISTRY:
            raise ValueError(
                f"Stage '{name}' is not registered in the v2 registry.  "
                f"Known stages: {sorted(STEP_REGISTRY)}."
            )

    return cfg


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class PreprocessingManager:
    """Orchestrates the v2 preprocessing pipeline based on the YAML config.

    Args:
        config_path: Path to ``configs/preprocessing_2.yaml``.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._cfg: Optional[dict] = None

    def run(self, stages_override: Optional[List[str]] = None) -> int:
        """Load config and execute the requested pipeline stages.

        Args:
            stages_override: If provided, only these stage names are run
                (``enabled`` flags in the YAML are ignored).

        Returns:
            Exit code: ``0`` on success, ``1`` on fatal error.
        """
        try:
            self._cfg = load_config(self._config_path)
        except Exception as exc:  # noqa: BLE001
            logger.critical("Failed to load config: %s", exc)
            return 1

        cfg = self._cfg

        logger.info("Global Preprocessing Pipeline v2")
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

        for stage_entry in stages_to_run:
            name     = stage_entry["name"]
            stage_id = stage_entry.get("id", "?")
            step_cls = STEP_REGISTRY[name]
            step     = step_cls()

            logger.info("")
            logger.info("=" * 70)
            logger.info("STAGE %s - %s", stage_id, name.upper())
            logger.info("=" * 70)

            t0 = time.perf_counter()
            try:
                step.run(cfg)
            except Exception as exc:  # noqa: BLE001
                logger.critical(
                    "Stage %s (%s) failed: %s", stage_id, name, exc,
                    exc_info=True,
                )
                return 1

            elapsed = time.perf_counter() - t0
            logger.info(
                "Stage %s (%s) finished in %.1f s.", stage_id, name, elapsed
            )

        total = time.perf_counter() - pipeline_start
        logger.info("")
        logger.info("=" * 70)
        logger.info(
            "Pipeline complete.  Stages: %s  |  Total time: %.1f s",
            [s["name"] for s in stages_to_run], total,
        )
        logger.info("=" * 70)
        return 0

    @staticmethod
    def _resolve_stages(
        stage_list: List[dict],
        override: Optional[List[str]],
    ) -> List[dict]:
        if override is not None:
            override_set = set(override)
            return [s for s in stage_list if s.get("name") in override_set]
        return [s for s in stage_list if s.get("enabled", True)]