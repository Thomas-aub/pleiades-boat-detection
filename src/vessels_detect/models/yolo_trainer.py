"""
src/vessels_detect/models/yolo_trainer.py
-----------------------------------------
Encapsulates all YOLO-OBB training logic in a single, testable class.

Replaces the flat script ``06_model.py`` and removes every hardcoded
hyperparameter.  All configuration is injected at construction time via a
:class:`~src.vessels_detect.utils.config.Config` object loaded from
``configs/train.yaml`` by the ``scripts/train.py`` entry-point.

Model loading modes
-------------------
The loading mode is selected automatically from ``model.weights`` in the
YAML - no code change is required when switching between model variants:

+------------------+-------------------------------------------+
| ``model.weights``| Behaviour                                 |
+==================+===========================================+
| ``*.pt``         | **Mode 3** - direct checkpoint load.      |
|                  | Standard fine-tuning; task is inferred    |
|                  | from the checkpoint.                      |
+------------------+-------------------------------------------+
| ``*.yaml``       | **Mode 2** - custom architecture.         |
|                  | Model is built from the YAML graph, then  |
|                  | ``model.load()`` transfers all layers     |
|                  | whose name *and* shape match              |
|                  | ``model.pretrained``.  New branches (e.g.|
|                  | a P2 detection head) train from random   |
|                  | init.  Set ``model.pretrained: ""`` to   |
|                  | train from scratch.                       |
+------------------+-------------------------------------------+
| ``training``     | **Mode 1** - resume from ``last.pt``      |
| ``.resume: true``| under ``project/run_name/weights/``.      |
+------------------+-------------------------------------------+

Two-stage training
------------------
When ``training.two_stage.enabled: true`` the trainer automatically runs
two sequential :meth:`model.train` calls:

1. **Stage 1** — backbone frozen (``stage1.freeze`` layers).
   The detection head and neck learn to predict with fixed backbone
   features.  This is fast and prevents the randomly-initialised head
   from corrupting the pretrained backbone weights early in training.

2. **Stage 2** — all layers unfrozen (``stage2.freeze: 0``).
   The best checkpoint from Stage 1 is reloaded and the full network is
   fine-tuned end-to-end with a reduced learning rate.

Each stage saves its outputs to a separate sub-directory:
``<project>/<run_name>_s1`` and ``<project>/<run_name>_s2``.

Design decisions
----------------
- **No global state.** The :class:`YoloTrainer` owns the model object and
  never stores results outside the instance.
- **Resume-safe.** If ``training.resume`` is ``True``, the trainer locates
  the last checkpoint automatically and raises :class:`FileNotFoundError`
  with an actionable message if it is absent.
- **Pure builder.** :meth:`_build_train_kwargs` has no side effects; all
  logging happens in the public methods that call it.
- **Augmentation isolation.** YOLO's native geometric augmentation pipeline
  is used for OBB tasks - albumentations geometric transforms cannot handle
  8-coordinate OBB labels correctly.
- **Logging only.** No ``print`` statements; all output goes through the
  standard ``logging`` module at appropriate levels.

Typical usage::

    from src.vessels_detect.utils.config import load_config
    from src.vessels_detect.models.yolo_trainer import YoloTrainer

    cfg     = load_config("configs/train.yaml")
    trainer = YoloTrainer(cfg)
    results = trainer.train()
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Optional

from src.vessels_detect.utils.config import Config

logger = logging.getLogger(__name__)


class YoloTrainer:
    """Initialises and trains a YOLO-OBB model from a :class:`Config`.

    The class separates *model initialisation* from *training execution*
    so that the model can be inspected or modified between the two steps
    in testing or interactive sessions.

    Supports both **single-stage** and **two-stage** training flows,
    controlled by ``training.two_stage.enabled`` in the YAML.

    Args:
        config: A :class:`~src.vessels_detect.utils.config.Config` loaded from
            ``configs/train.yaml``.  The expected top-level keys are
            ``model``, ``training``, and ``augmentation``.

    Raises:
        ImportError: If the ``ultralytics`` package is not installed.
    """

    def __init__(self, config: Config) -> None:
        try:
            from ultralytics import YOLO  # noqa: F401 - validate at init time
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required for YoloTrainer. "
                "Install it with:  pip install ultralytics"
            ) from exc

        self._cfg: Config          = config
        self._model: Optional[Any] = None  # ultralytics.YOLO, typed as Any for mypy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load or create the YOLO model according to ``configs/train.yaml``.

        See the module-level docstring for the full loading mode table.

        Raises:
            FileNotFoundError: In any of the following situations:

                - ``training.resume: true`` but ``last.pt`` is absent.
                - ``model.weights`` is a ``.yaml`` that does not exist.
                - ``model.pretrained`` is set but the ``.pt`` does not exist.
                - ``model.weights`` is a ``.pt`` path that does not exist.
        """
        from ultralytics import YOLO

        model_cfg = self._cfg.model
        train_cfg = self._cfg.training
        resume    = bool(train_cfg.get("resume", False))

        # ── Mode 1: resume from last checkpoint ───────────────────────────
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
            return

        weights = str(model_cfg.weights)

        # ── Mode 2: custom architecture YAML + optional weight transfer ───
        # Triggered when model.weights is a .yaml file (e.g. a P2 variant or
        # any other structural modification of the base graph).
        # YOLO(yaml, task=...) builds the full compute graph from scratch;
        # model.load(pt) then copies all parameters whose name AND shape match
        # the pretrained checkpoint.  Unmatched layers (new branches such as a
        # P2 detection head) remain at random initialisation - this is expected
        # and correct.
        if weights.endswith(".yaml"):
            yaml_path = Path(weights)
            if not yaml_path.exists():
                raise FileNotFoundError(
                    f"Architecture YAML not found: '{yaml_path}'. "
                    "Check model.weights in train.yaml."
                )

            # task must be provided explicitly when loading from a .yaml because
            # there is no checkpoint metadata to infer it from.
            task = str(train_cfg.get("task", "obb"))
            logger.info(
                "Building model from architecture YAML: %s (task=%s)",
                yaml_path.name, task,
            )
            self._model = YOLO(str(yaml_path), task=task)

            pretrained = str(model_cfg.get("pretrained", ""))
            if pretrained:
                pt_path = Path(pretrained)
                if not pt_path.exists():
                    raise FileNotFoundError(
                        f"Pretrained weights not found: '{pt_path}'. "
                        "Download the base checkpoint or set "
                        "model.pretrained: '' to train from scratch."
                    )
                logger.info(
                    "Transferring pretrained weights: %s → %s",
                    pt_path.name, yaml_path.name,
                )
                # model.load() is Ultralytics' partial-transfer method.
                # Layers are matched by parameter name and tensor shape;
                # unmatched layers retain their random initialisation.
                self._model.load(str(pt_path))
                logger.info("Weight transfer complete.")
            else:
                logger.warning(
                    "model.pretrained is empty - training custom architecture "
                    "entirely from scratch.  Set model.pretrained: <path>.pt "
                    "for transfer learning."
                )
            return

        # ── Mode 3: direct .pt checkpoint ─────────────────────────────────
        pt_path = Path(weights)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Weights not found: '{pt_path}'. "
                "Provide a valid .pt checkpoint or a .yaml architecture file "
                "under model.weights."
            )
        logger.info("Loading checkpoint: %s", weights)
        self._model = YOLO(weights)

    def train(self) -> Any:
        """Execute YOLO training with the hyperparameters from the config.

        Automatically selects single-stage or two-stage training based on
        ``training.two_stage.enabled`` in the YAML.

        Calls :meth:`load_model` automatically if the model has not been
        loaded yet.

        Returns:
            The ``ultralytics`` training results object from the final stage.
            Key attributes:

            - ``results.box.map50``  - mAP@50 on the validation set.
            - ``results.box.map``    - mAP@50-95.
            - ``results.save_dir``   - directory containing all outputs.

        Raises:
            FileNotFoundError: Propagated from :meth:`load_model`, or raised
                when the Stage 1 ``best.pt`` is missing before Stage 2 starts.
            RuntimeError: If training fails (propagated from ``ultralytics``).
        """
        if self._model is None:
            self.load_model()

        train_cfg     = self._cfg.training
        aug_cfg       = self._cfg.augmentation
        two_stage_cfg = train_cfg.get("two_stage", {})

        if two_stage_cfg and bool(two_stage_cfg.get("enabled", False)):
            return self._train_two_stage(train_cfg, aug_cfg, two_stage_cfg)

        return self._train_single_stage(train_cfg, aug_cfg)

    @property
    def model(self) -> Any:
        """The underlying ``ultralytics.YOLO`` instance.

        Returns:
            The loaded YOLO model, or ``None`` if :meth:`load_model` has
            not been called yet.
        """
        return self._model

    # ------------------------------------------------------------------
    # Private — training orchestration
    # ------------------------------------------------------------------

    def _train_single_stage(
        self,
        train_cfg: Config,
        aug_cfg: Config,
    ) -> Any:
        """Run a single YOLO training call (classic behaviour).

        Args:
            train_cfg: The ``training`` sub-section of the config.
            aug_cfg:   The ``augmentation`` sub-section of the config.

        Returns:
            The ``ultralytics`` training results object.
        """
        train_kwargs = self._build_train_kwargs(train_cfg, aug_cfg)
        self._log_stage_summary(train_kwargs, stage_label="Single-stage")
        results = self._model.train(**train_kwargs)
        self._log_training_complete(train_kwargs)
        return results

    def _train_two_stage(
        self,
        train_cfg: Config,
        aug_cfg: Config,
        two_stage_cfg: Any,
    ) -> Any:
        """Run the two-stage frozen → unfrozen training pipeline.

        Stage 1 freezes the first N backbone layers so that the detection
        head and neck learn stable features before the backbone is allowed
        to drift.  Stage 2 reloads the best Stage 1 checkpoint and
        fine-tunes all layers end-to-end with a reduced learning rate.

        Args:
            train_cfg:     The ``training`` sub-section of the config.
            aug_cfg:       The ``augmentation`` sub-section of the config.
            two_stage_cfg: The ``training.two_stage`` mapping from the YAML.

        Returns:
            The ``ultralytics`` training results object from Stage 2.

        Raises:
            FileNotFoundError: If the Stage 1 ``best.pt`` is missing when
                Stage 2 is about to start.
        """
        from ultralytics import YOLO

        base_kwargs = self._build_train_kwargs(train_cfg, aug_cfg)
        base_name   = base_kwargs["name"]   # e.g. "pleiade_hr"
        project     = base_kwargs["project"]

        # ── Stage 1: frozen backbone ───────────────────────────────────────
        s1_raw    = two_stage_cfg.get("stage1", {})
        s1_kwargs = self._apply_stage_overrides(
            base_kwargs,
            stage_cfg=s1_raw,
            default_name=f"{base_name}_s1",
        )

        logger.info("=" * 70)
        logger.info("TWO-STAGE TRAINING  —  Stage 1 / 2  (frozen backbone)")
        logger.info("=" * 70)
        self._log_stage_summary(s1_kwargs, stage_label="Stage 1")
        self._model.train(**s1_kwargs)

        # Reload best checkpoint produced by Stage 1.
        s1_best = Path(project) / s1_kwargs["name"] / "weights" / "best.pt"
        if not s1_best.exists():
            raise FileNotFoundError(
                f"Stage 1 best checkpoint not found at '{s1_best}'. "
                "Ensure Stage 1 completed successfully and save=True."
            )
        logger.info("Stage 1 complete.  Loading best checkpoint: %s", s1_best)
        self._model = YOLO(str(s1_best))

        # ── Stage 2: full fine-tuning ──────────────────────────────────────
        s2_raw    = two_stage_cfg.get("stage2", {})
        s2_kwargs = self._apply_stage_overrides(
            base_kwargs,
            stage_cfg=s2_raw,
            default_name=f"{base_name}_s2",
        )

        logger.info("=" * 70)
        logger.info("TWO-STAGE TRAINING  —  Stage 2 / 2  (full fine-tuning)")
        logger.info("=" * 70)
        self._log_stage_summary(s2_kwargs, stage_label="Stage 2")
        results = self._model.train(**s2_kwargs)

        self._log_training_complete(s2_kwargs)
        return results

    # ------------------------------------------------------------------
    # Private — kwargs builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_train_kwargs(train_cfg: Config, aug_cfg: Config) -> dict:
        """Assemble the ``model.train()`` keyword-argument dictionary.

        This method is intentionally **pure** - it has no side effects and
        performs no logging.  All values are read from the config with
        explicit fallback defaults so that a partial YAML still produces a
        valid training call.

        Two parameters are injected conditionally rather than always:

        - ``freeze`` - omitted when ``0`` to maintain compatibility with
          older Ultralytics releases that do not accept the parameter.
        - ``copy_paste`` - omitted when ``0.0``; the OBB copy-paste
          augmentation is only active at values > 0.

        Args:
            train_cfg: The ``training`` sub-section of the config.
            aug_cfg: The ``augmentation`` sub-section of the config.

        Returns:
            A flat dictionary ready to be unpacked into ``model.train(**...)``.
        """
        def _f(cfg: Config, key: str, default: Any) -> Any:
            """Safe getter with explicit default."""
            return cfg.get(key, default)

        kwargs: dict = {
            # ── Dataset ───────────────────────────────────────────────────
            "data": str(_f(train_cfg, "dataset_yaml", "data/dataset.yaml")),

            # ── Core training ─────────────────────────────────────────────
            "epochs":      int(_f(train_cfg,   "epochs",      200)),
            "imgsz":       int(_f(train_cfg,   "imgsz",       640)),
            "batch":       int(_f(train_cfg,   "batch_size",  8)),
            "workers":     int(_f(train_cfg,   "workers",     4)),
            "patience":    int(_f(train_cfg,   "patience",    0)),
            "save_period": int(_f(train_cfg,   "save_period", 30)),
            "seed":        int(_f(train_cfg,   "seed",        0)),
            "amp":         bool(_f(train_cfg,  "amp",         True)),
            "cache":       _f(train_cfg,       "cache",       False),

            # ── Optimiser ─────────────────────────────────────────────────
            "optimizer":       str(_f(train_cfg,   "optimizer",       "auto")),
            "lr0":             float(_f(train_cfg, "lr0",             0.005)),
            "lrf":             float(_f(train_cfg, "lrf",             0.1)),
            "momentum":        float(_f(train_cfg, "momentum",        0.937)),
            "weight_decay":    float(_f(train_cfg, "weight_decay",    0.0005)),
            "warmup_epochs":   float(_f(train_cfg, "warmup_epochs",   5.0)),
            "warmup_momentum": float(_f(train_cfg, "warmup_momentum", 0.8)),
            "warmup_bias_lr":  float(_f(train_cfg, "warmup_bias_lr",  0.1)),
            "nbs":             int(_f(train_cfg,   "nbs",             64)),
            "cos_lr":          bool(_f(train_cfg,  "cos_lr",          False)),

            # ── Regularisation ────────────────────────────────────────────
            "label_smoothing": float(_f(train_cfg, "label_smoothing", 0.0)),

            # ── Run management ────────────────────────────────────────────
            "project": str(_f(train_cfg,  "project",     "runs/obb")),
            "name":    str(_f(train_cfg,  "run_name",    "boat_obb")),
            "device":  _f(train_cfg,      "device",      0),
            "resume":  bool(_f(train_cfg, "resume",      False)),
            "verbose": bool(_f(train_cfg, "verbose",     False)),
            "plots":   bool(_f(train_cfg, "plots",       True)),

            # ── Augmentation - photometric ────────────────────────────────
            "augment": True,
            "hsv_h":   float(_f(aug_cfg, "hsv_h", 0.015)),
            "hsv_s":   float(_f(aug_cfg, "hsv_s", 0.4)),
            "hsv_v":   float(_f(aug_cfg, "hsv_v", 0.3)),

            # ── Augmentation - geometric (YOLO native; OBB-label-aware) ───
            "mosaic":       float(_f(aug_cfg, "mosaic",       0.5)),
            "close_mosaic": int(_f(aug_cfg,   "close_mosaic", 20)),
            "degrees":      float(_f(aug_cfg, "degrees",      180.0)),
            "fliplr":       float(_f(aug_cfg, "fliplr",       0.5)),
            "flipud":       float(_f(aug_cfg, "flipud",       0.5)),
            "scale":        float(_f(aug_cfg, "scale",        0.3)),
            "multi_scale":  float(_f(aug_cfg, "multi_scale",  0.0)),
            "perspective":  float(_f(aug_cfg, "perspective",  0.0)),
        }

        # ── Conditional injection ─────────────────────────────────────────
        # freeze=0 is omitted for compatibility with older Ultralytics builds
        # that do not accept the parameter.
        freeze = int(_f(train_cfg, "freeze", 0))
        if freeze > 0:
            kwargs["freeze"] = freeze

        # copy_paste is omitted when disabled (value = 0.0).
        copy_paste = float(_f(aug_cfg, "copy_paste", 0.0))
        if copy_paste > 0.0:
            kwargs["copy_paste"] = copy_paste

        return kwargs

    @staticmethod
    def _apply_stage_overrides(
        base_kwargs: dict,
        stage_cfg: Any,
        default_name: str,
    ) -> dict:
        """Return a copy of *base_kwargs* with stage-specific overrides applied.

        Only keys present in *stage_cfg* are overridden, so the caller only
        needs to specify what is different for that stage.

        The ``exist_ok`` flag is always set to ``True`` so that re-running a
        stage does not fail because the output directory already exists.

        Args:
            base_kwargs:  The full kwargs dict produced by
                :meth:`_build_train_kwargs`.
            stage_cfg:    The ``stage1`` or ``stage2`` mapping from the YAML
                (may be an empty dict if the key is absent).
            default_name: Fallback ``name`` if the stage config does not
                specify ``run_name``.  Typically ``<base_name>_s1`` or
                ``<base_name>_s2``.

        Returns:
            A new dict with stage-specific values applied.
        """
        kwargs = copy.copy(base_kwargs)

        # Always use stage-specific output directory to keep results separate.
        kwargs["name"] = default_name
        # Allow re-running without manually deleting previous stage output.
        kwargs["exist_ok"] = True
        # Never resume inside a stage — each stage starts fresh from its
        # checkpoint (either the pretrained weights or Stage 1 best.pt).
        kwargs["resume"] = False

        if not stage_cfg:
            return kwargs

        # Per-stage overrides — only applied when the key is present.
        _int_keys   = {"epochs": "epochs", "close_mosaic": "close_mosaic"}
        _float_keys = {"lr0": "lr0", "lrf": "lrf"}
        _str_keys   = {"run_name": "name"}

        for src_key, dst_key in _int_keys.items():
            if src_key in stage_cfg:
                kwargs[dst_key] = int(stage_cfg.get(src_key))

        for src_key, dst_key in _float_keys.items():
            if src_key in stage_cfg:
                kwargs[dst_key] = float(stage_cfg.get(src_key))

        for src_key, dst_key in _str_keys.items():
            if src_key in stage_cfg:
                kwargs[dst_key] = str(stage_cfg.get(src_key))

        # Freeze is handled separately: 0 means "remove the key entirely"
        # so that Ultralytics unfreezes all layers rather than freezing none.
        if "freeze" in stage_cfg:
            freeze = int(stage_cfg.get("freeze"))
            if freeze > 0:
                kwargs["freeze"] = freeze
            else:
                kwargs.pop("freeze", None)   # 0 → unfreeze all layers

        return kwargs

    # ------------------------------------------------------------------
    # Private — logging helpers
    # ------------------------------------------------------------------

    def _log_stage_summary(self, train_kwargs: dict, stage_label: str) -> None:
        """Emit a structured summary of the training kwargs to the log.

        Args:
            train_kwargs: The resolved kwargs dict for this stage.
            stage_label:  Human-readable label, e.g. ``"Stage 1"`` or
                          ``"Single-stage"``.
        """
        freeze = train_kwargs.get("freeze", 0)
        logger.info("YOLO-OBB Training — %s — %s", stage_label, train_kwargs.get("name", ""))
        logger.info("  Weights     : %s", self._cfg.model.weights)
        logger.info("  Dataset     : %s", train_kwargs["data"])
        logger.info("  Epochs      : %d", train_kwargs["epochs"])
        logger.info("  Img size    : %d px", train_kwargs["imgsz"])
        logger.info("  Batch       : %d", train_kwargs["batch"])
        logger.info("  Device      : %s", train_kwargs["device"])
        logger.info("  Optimizer   : %s", train_kwargs.get("optimizer", "auto"))
        logger.info(
            "  LR (lr0/lrf): %.4f / %.4f",
            train_kwargs["lr0"], train_kwargs["lrf"],
        )
        logger.info("  Warmup eps  : %.1f", train_kwargs["warmup_epochs"])
        logger.info("  AMP         : %s", train_kwargs.get("amp", True))
        logger.info("  Cache       : %s", train_kwargs.get("cache", False))
        logger.info("  Freeze      : %s layer(s)", freeze if freeze else "none (all trainable)")
        logger.info(
            "  Run         : %s / %s",
            train_kwargs.get("project", ""), train_kwargs.get("name", ""),
        )

    @staticmethod
    def _log_training_complete(train_kwargs: dict) -> None:
        """Log the location of the best checkpoint after training finishes.

        Args:
            train_kwargs: The resolved kwargs dict for the final stage.
        """
        best_ckpt = (
            Path(str(train_kwargs.get("project", "")))
            / str(train_kwargs.get("name", ""))
            / "weights"
            / "best.pt"
        )
        logger.info("Training complete.")
        logger.info("  Best checkpoint : %s", best_ckpt)
        logger.info("  Results dir     : %s", best_ckpt.parent.parent)
