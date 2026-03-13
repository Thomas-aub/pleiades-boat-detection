"""
src/models/yolo_trainer.py
---------------------------
Encapsulates all YOLO-OBB training logic in a single, testable class.

Replaces the flat script ``06_model.py`` and removes every hardcoded
hyperparameter.  All configuration is injected at construction time via a
:class:`~src.utils.config.Config` object that is loaded from
``configs/train.yaml`` by the ``scripts/train.py`` entry-point.

Design decisions
----------------
- **No global state.** The :class:`YoloTrainer` owns the model object and
  never stores results outside the instance.
- **Resume-safe.** If ``training.resume`` is ``True``, the trainer locates
  the last checkpoint automatically and raises :class:`FileNotFoundError`
  with an actionable message if it is absent.
- **Augmentation isolation.** YOLO's native geometric augmentation pipeline
  is used for OBB tasks (albumentations geometric transforms cannot handle
  8-coordinate OBB labels correctly).
- **Logging only.** No ``print`` statements; all output goes through the
  standard ``logging`` module at appropriate levels.

Typical usage::

    from src.utils.config import load_config
    from src.models.yolo_trainer import YoloTrainer

    cfg     = load_config("configs/train.yaml")
    trainer = YoloTrainer(cfg)
    results = trainer.train()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.utils.config import Config

logger = logging.getLogger(__name__)


class YoloTrainer:
    """Initialises and trains a YOLO-OBB model from a :class:`Config`.

    The class separates *model initialisation* from *training execution*
    so that the model can be inspected or modified between the two steps
    in testing or interactive sessions.

    Args:
        config: A :class:`~src.utils.config.Config` loaded from
            ``configs/train.yaml``.  The expected structure is described
            in the YAML file itself.

    Raises:
        ImportError: If the ``ultralytics`` package is not installed.
    """

    def __init__(self, config: Config) -> None:
        try:
            from ultralytics import YOLO  # noqa: F401 — validate at init time
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for YoloTrainer. "
                "Install it with:  pip install ultralytics"
            ) from exc

        self._cfg   = config
        self._model: Optional[Any] = None  # ultralytics.YOLO, typed as Any for mypy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load or create the YOLO model from the configured weights path.

        If ``training.resume`` is ``True``, the trainer searches for the
        ``last.pt`` checkpoint under the configured project + run-name
        directory.  If ``False`` (default), the pretrained hub weights are
        loaded (downloaded automatically on first run).

        Raises:
            FileNotFoundError: If ``resume=True`` but no ``last.pt`` exists.
        """
        from ultralytics import YOLO

        model_cfg  = self._cfg.model
        train_cfg  = self._cfg.training
        resume     = bool(train_cfg.get("resume", False))

        if resume:
            project   = str(train_cfg.get("project", "runs/obb"))
            run_name  = str(train_cfg.get("run_name", "boat_obb"))
            last_ckpt = Path(project) / run_name / "weights" / "last.pt"

            if not last_ckpt.exists():
                raise FileNotFoundError(
                    f"Cannot resume: checkpoint not found at '{last_ckpt}'. "
                    "Set training.resume: false to start a new run."
                )
            logger.info("Resuming from checkpoint: %s", last_ckpt)
            self._model = YOLO(str(last_ckpt))
        else:
            weights = str(model_cfg.weights)
            logger.info("Loading model weights: %s", weights)
            self._model = YOLO(weights)

    def train(self) -> Any:
        """Execute YOLO training with the hyperparameters from the config.

        Calls :meth:`load_model` automatically if the model has not been
        loaded yet.

        Returns:
            The ``ultralytics`` training results object.  Key attributes:

            - ``results.box.map50``   — mAP@50 on the validation set.
            - ``results.box.map``     — mAP@50-95.
            - ``results.save_dir``    — directory containing all outputs.

        Raises:
            RuntimeError: If training fails for any reason (propagated from
                ``ultralytics``).
        """
        if self._model is None:
            self.load_model()

        train_cfg = self._cfg.training
        aug_cfg   = self._cfg.augmentation

        # Build the full kwargs dict so it can be logged cleanly.
        train_kwargs = self._build_train_kwargs(train_cfg, aug_cfg)

        logger.info("=" * 70)
        logger.info("YOLO-OBB Training — %s", train_kwargs.get("name", "run"))
        logger.info("=" * 70)
        logger.info("  Weights  : %s", self._cfg.model.weights)
        logger.info("  Dataset  : %s", train_kwargs["data"])
        logger.info("  Epochs   : %s", train_kwargs["epochs"])
        logger.info("  Img size : %s", train_kwargs["imgsz"])
        logger.info("  Batch    : %s", train_kwargs["batch"])
        logger.info("  Device   : %s", train_kwargs["device"])
        logger.info("  Run      : %s / %s", train_kwargs.get("project", ""), train_kwargs.get("name", ""))

        results = self._model.train(**train_kwargs)

        best_ckpt = (
            Path(str(train_kwargs.get("project", "")))
            / str(train_kwargs.get("name", ""))
            / "weights"
            / "best.pt"
        )
        logger.info("Training complete.")
        logger.info("  Best checkpoint : %s", best_ckpt)
        logger.info("  Results dir     : %s", best_ckpt.parent.parent)

        return results

    @property
    def model(self) -> Any:
        """The underlying ``ultralytics.YOLO`` instance (None before load).

        Returns:
            The loaded YOLO model, or ``None`` if :meth:`load_model` has
            not been called.
        """
        return self._model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_train_kwargs(train_cfg: Config, aug_cfg: Config) -> dict:
        """Assemble the ``model.train()`` keyword-argument dictionary.

        All values are read from the config with explicit fallback defaults
        so that a partial YAML still produces a valid training call.

        Args:
            train_cfg: The ``training`` sub-section of the config.
            aug_cfg: The ``augmentation`` sub-section of the config.

        Returns:
            A flat dictionary ready to be unpacked into ``model.train(**...)``.
        """
        def _f(cfg: Config, key: str, default: Any) -> Any:
            """Safe getter with default."""
            return cfg.get(key, default)

        return {
            # Dataset
            "data":          str(_f(train_cfg, "dataset_yaml", "data/dataset.yaml")),

            # Core training
            "epochs":        int(_f(train_cfg, "epochs", 200)),
            "imgsz":         int(_f(train_cfg, "imgsz", 640)),
            "batch":         int(_f(train_cfg, "batch_size", 8)),
            "workers":       int(_f(train_cfg, "workers", 4)),
            "patience":      int(_f(train_cfg, "patience", 15)),
            "save_period":   int(_f(train_cfg, "save_period", 30)),

            # Optimiser
            "lr0":           float(_f(train_cfg, "lr0", 0.005)),
            "lrf":           float(_f(train_cfg, "lrf", 0.1)),
            "momentum":      float(_f(train_cfg, "momentum", 0.937)),
            "weight_decay":  float(_f(train_cfg, "weight_decay", 0.0005)),
            "warmup_epochs": float(_f(train_cfg, "warmup_epochs", 3.0)),

            # Run management
            "project":       str(_f(train_cfg, "project", "runs/obb")),
            "name":          str(_f(train_cfg, "run_name", "boat_obb")),
            "device":        _f(train_cfg, "device", 0),
            "resume":        bool(_f(train_cfg, "resume", False)),
            "verbose":       bool(_f(train_cfg, "verbose", False)),

            # Augmentation — photometric (YOLO native; safe for OBB)
            "augment":       True,
            "hsv_h":         float(_f(aug_cfg, "hsv_h", 0.015)),
            "hsv_s":         float(_f(aug_cfg, "hsv_s", 0.4)),
            "hsv_v":         float(_f(aug_cfg, "hsv_v", 0.3)),

            # Augmentation — geometric (YOLO native; OBB-label-aware)
            "mosaic":        float(_f(aug_cfg, "mosaic", 0.5)),
            "close_mosaic":  int(_f(aug_cfg, "close_mosaic", 20)),
            "degrees":       float(_f(aug_cfg, "degrees", 180.0)),
            "fliplr":        float(_f(aug_cfg, "fliplr", 0.5)),
            "flipud":        float(_f(aug_cfg, "flipud", 0.5)),
            "scale":         float(_f(aug_cfg, "scale", 0.0)),
            "multi_scale":   float(_f(aug_cfg, "multi_scale", 0.0)),
            "perspective":   float(_f(aug_cfg, "perspective", 0.0)),
        }
