"""
src/vessels_detect/manager.py
------------------------------
Top-level orchestrator for the full prediction pipeline.

This manager sits above the sub-pipeline managers
(:class:`~src.vessels_detect.preprocessing.manager.PreprocessingManager` and
:class:`~src.vessels_detect.postprocessing.manager.PostprocessingManager`) and
adds the two stages that are unique to prediction: running the YOLO model on
tiles and merging tile-level detections into per-image GeoJSON files.

Two modes
~~~~~~~~~
``inference``
    preprocess → predict → postprocess.

``evaluation``
    preprocess → predict → postprocess → evaluate (match against GT, compute
    metrics, optionally write labelled GeoJSONs).

Architecture
~~~~~~~~~~~~
::

    manager.py  (this file)
    │
    ├── load_config()              - parse predict.yaml, resolve all paths
    │
    └── PredictManager
            run()
              ├── [optional] PreprocessingManager.run()
              ├── Predictor.run()          → predictions_dir/  (raw GeoJSONs)
              ├── [optional] PostprocessingManager.run()
              └── [evaluation] Evaluator.run()

Each sub-component is a separate class with a single ``run(cfg)`` method.
The top-level config dict is threaded through every stage unchanged so that
cross-stage paths (e.g. ``predictions_dir`` consumed by the evaluator) are
always resolved from a single source of truth.

Typical usage::

    from src.vessels_detect.manager import PredictManager

    manager = PredictManager(config_path=Path("configs/predict.yaml"))
    sys.exit(manager.run())
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import yaml

from src.vessels_detect.predict.predictor      import Predictor
from src.vessels_detect.predict.evaluation     import Evaluator
from src.vessels_detect.preprocessing.manager  import PreprocessingManager
from src.vessels_detect.postprocessing.manager import PostprocessingManager

logger = logging.getLogger(__name__)

_VALID_MODES = {"inference", "evaluation"}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load, validate, and resolve the predict.yaml configuration.

    All path values under ``cfg["paths"]`` are resolved to absolute
    :class:`~pathlib.Path` objects.  Sub-pipeline config file paths
    (``preprocessing.config``, ``postprocessing.config``) are also
    resolved.

    Args:
        config_path: Path to ``configs/predict.yaml``.

    Returns:
        Fully resolved configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        yaml.YAMLError:    If the file is not valid YAML.
        KeyError:          If a required top-level key is missing.
        ValueError:        If ``pipeline.mode`` is not a recognised value.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh)

    for section in ("pipeline", "model", "paths", "prediction"):
        if section not in cfg:
            raise KeyError(f"Required section '{section}' missing from '{config_path}'.")

    mode = cfg["pipeline"].get("mode", "")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"pipeline.mode must be one of {sorted(_VALID_MODES)}, got '{mode}'."
        )

    # Resolve all I/O paths.
    cfg["paths"] = {k: Path(v).resolve() for k, v in cfg["paths"].items()}

    # Resolve sub-pipeline config paths.
    for sub in ("preprocessing", "postprocessing"):
        if sub in cfg and cfg[sub].get("config"):
            cfg[sub]["config"] = Path(cfg[sub]["config"]).resolve()

    return cfg


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class PredictManager:
    """Orchestrates the full prediction pipeline from a single YAML config.

    Args:
        config_path: Path to ``configs/predict.yaml``.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._cfg: Optional[dict] = None

    def run(self) -> int:
        """Execute the full pipeline according to ``predict.yaml``.

        Returns:
            Exit code: ``0`` on success, ``1`` on fatal error.
        """
        try:
            self._cfg = load_config(self._config_path)
        except Exception as exc:  # noqa: BLE001
            logger.critical("Failed to load predict config: %s", exc)
            return 1

        cfg  = self._cfg
        mode = cfg["pipeline"]["mode"]

        logger.info("=" * 70)
        logger.info("Prediction Pipeline  [mode: %s]", mode.upper())
        logger.info("Config : %s", self._config_path.resolve())
        logger.info("=" * 70)
        self._log_paths(cfg)

        t_start = time.perf_counter()

        # ── 1. Preprocessing ──────────────────────────────────────────────
        if cfg.get("preprocessing", {}).get("enabled", False):
            rc = self._run_preprocessing(cfg)
            if rc != 0:
                return rc
        else:
            logger.info("Preprocessing: skipped (enabled: false).")

        # ── 2. Prediction ─────────────────────────────────────────────────
        rc = self._run_prediction(cfg)
        if rc != 0:
            return rc

        # ── 3. Postprocessing ─────────────────────────────────────────────
        if cfg.get("postprocessing", {}).get("enabled", False):
            rc = self._run_postprocessing(cfg)
            if rc != 0:
                return rc
        else:
            logger.info("Postprocessing: skipped (enabled: false).")

        # ── 4. Evaluation (evaluation mode only) ──────────────────────────
        if mode == "evaluation":
            rc = self._run_evaluation(cfg)
            if rc != 0:
                return rc

        elapsed = time.perf_counter() - t_start
        logger.info("")
        logger.info("=" * 70)
        logger.info("Pipeline complete  [mode: %s]  |  Total: %.1f s", mode, elapsed)
        logger.info("=" * 70)
        return 0

    # ------------------------------------------------------------------
    # Private stage runners
    # ------------------------------------------------------------------

    @staticmethod
    def _run_preprocessing(cfg: dict) -> int:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STAGE - PREPROCESSING")
        logger.info("=" * 70)
        sub_cfg_path: Path = cfg["preprocessing"]["config"]
        manager = PreprocessingManager(config_path=sub_cfg_path)
        return manager.run()

    @staticmethod
    def _run_prediction(cfg: dict) -> int:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STAGE - PREDICTION")
        logger.info("=" * 70)
        predictor = Predictor()
        return predictor.run(cfg)

    @staticmethod
    def _run_postprocessing(cfg: dict) -> int:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STAGE - POSTPROCESSING")
        logger.info("=" * 70)
        sub_cfg_path: Path = cfg["postprocessing"]["config"]
        manager = PostprocessingManager(config_path=sub_cfg_path)
        return manager.run()

    @staticmethod
    def _run_evaluation(cfg: dict) -> int:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STAGE - EVALUATION")
        logger.info("=" * 70)
        evaluator = Evaluator()
        return evaluator.run(cfg)

    @staticmethod
    def _log_paths(cfg: dict) -> None:
        logger.info("Paths:")
        for key, val in cfg["paths"].items():
            logger.info("  %-22s: %s", key, val)
