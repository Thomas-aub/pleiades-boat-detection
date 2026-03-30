"""
src/vessels_detect/utils/notebook_helpers.py
---------------------------------------------
Utility functions for building pipeline objects directly from explicit
parameters, bypassing YAML config files.

These helpers are intended to be called exclusively from notebooks.
All business logic lives in the core modules; this file only provides
thin, typed adapters so notebooks stay free of infrastructure boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Config-compatible shim
# ---------------------------------------------------------------------------

class _DotDict(dict):
    """A dict subclass that supports attribute-style access.

    Mirrors the interface expected by :class:`~src.vessels_detect.utils.config.Config`
    so that :class:`~src.vessels_detect.inference.predictor.YoloPredictor` can
    read ``cfg.model.weights`` and ``cfg.inference.get("conf", 0.10)`` without
    loading a YAML file.
    """

    def __getattr__(self, name: str):
        try:
            val = self[name]
        except KeyError:
            raise AttributeError(name) from None
        if isinstance(val, dict):
            return _DotDict(val)
        return val

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


def build_predictor_config(
    weights: Path,
    conf: float = 0.10,
    iou: float = 0.30,
    imgsz: int = 640,
) -> "_DotDict":
    """Build a Config-compatible object for :class:`~src.vessels_detect.inference.predictor.YoloPredictor`.

    Args:
        weights: Path to the YOLO ``.pt`` checkpoint.
        conf: Confidence threshold; detections below are dropped.
        iou: YOLO-internal NMS IoU threshold.
        imgsz: Inference image size in pixels.

    Returns:
        A :class:`_DotDict` that satisfies the ``Config`` interface
        consumed by :class:`~src.vessels_detect.inference.predictor.YoloPredictor`.

    Example::

        config = build_predictor_config(
            weights = Path("boat_obb_v32/weights/best.pt"),
            conf    = 0.10,
            iou     = 0.30,
            imgsz   = 640,
        )
        predictor = YoloPredictor(config=config)
    """
    return _DotDict({
        "model": {
            "weights": str(weights),
        },
        "inference": {
            "conf":  conf,
            "iou":   iou,
            "imgsz": imgsz,
        },
    })
