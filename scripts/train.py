#!/usr/bin/env python3
"""
scripts/train.py
-----------------
Entry-point for YOLO-OBB boat-detection model training.

Reads ``configs/train.yaml``, instantiates
:class:`~src.vessels_detect.models.yolo_trainer.YoloTrainer`, and starts
the training run.  All hyperparameters are driven by the YAML; no values
are hardcoded here.

Model variant is controlled exclusively by ``model.weights`` in the YAML:

- ``*.pt``   - standard fine-tuning from a pretrained checkpoint.
- ``*.yaml`` - custom architecture (e.g. P2 head) with optional partial
  weight transfer from ``model.pretrained``.

Usage::

    PYTHONPATH=. python scripts/train.py
    PYTHONPATH=. python scripts/train.py --config configs/train.yaml
    PYTHONPATH=. python scripts/train.py --config configs/train.yaml --log-level DEBUG

    # Override individual keys without editing the YAML (dot-separated path):
    PYTHONPATH=. python scripts/train.py --set training.epochs=50 training.batch_size=4
    PYTHONPATH=. python scripts/train.py --set model.weights=weights/yolo26m-obb.pt

Suppress OpenCV log spam::

    OPENCV_LOG_LEVEL=ERROR PYTHONPATH=. python scripts/train.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from src.vessels_detect.utils.config import load_config
from src.vessels_detect.models.yolo_trainer import YoloTrainer

import os
# Force OpenCV à se taire et à ignorer les warnings inoffensifs
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level_name: str) -> None:
    """Set up root logging with a timestamped format.

    Args:
        level_name: Logging level string (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Suppress per-batch ultralytics output; surface only warnings and above.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Train the YOLO-OBB boat detection model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the training config YAML (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=(
            "Override config values without editing the YAML.  "
            "Use dot-separated paths, e.g.  "
            "--set training.epochs=100 model.weights=weights/yolo26m-obb.pt"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def _parse_overrides(raw: List[str]) -> dict:
    """Convert ``["a.b=1", "c=foo"]`` to a nested override dictionary.

    Numeric strings are coerced to ``int`` then ``float`` before falling back
    to ``str``.  The boolean literals ``"true"`` and ``"false"`` (case-
    insensitive) are also coerced to their Python equivalents.

    Args:
        raw: List of ``key=value`` strings where the key may contain dots
            to indicate nesting (e.g. ``"training.epochs=50"``).

    Returns:
        A nested dictionary suitable for
        :meth:`~src.vessels_detect.utils.config.Config.merge`.
    """
    overrides: dict = {}
    for item in raw:
        if "=" not in item:
            logger.warning("Ignoring malformed override '%s' (missing '=').", item)
            continue

        key_path, _, raw_val = item.partition("=")

        # Type coercion: bool → int → float → str.
        if raw_val.lower() == "true":
            value: Any = True
        elif raw_val.lower() == "false":
            value = False
        else:
            try:
                value = int(raw_val)
            except ValueError:
                try:
                    value = float(raw_val)
                except ValueError:
                    value = raw_val

        # Build nested dict from dotted path.
        parts   = key_path.split(".")
        current = overrides
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    return overrides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Load config, optionally apply CLI overrides, and start training.

    Args:
        argv: Optional argument list (for testing).

    Returns:
        Exit code: ``0`` on success, ``1`` on failure.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    logger.info("Boat Detection - YOLO-OBB Training")
    logger.info("Config: %s", args.config.resolve())

    # ── Load config ────────────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        logger.critical("Failed to load config: %s", exc)
        return 1

    if args.set:
        overrides = _parse_overrides(args.set)
        cfg = cfg.merge(overrides)
        logger.info("Applied CLI overrides: %s", overrides)

    # ── Train ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        trainer = YoloTrainer(cfg)
        trainer.train()
    except FileNotFoundError as exc:
        logger.critical("File not found: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.critical("Training failed: %s", exc, exc_info=True)
        return 1

    elapsed = time.perf_counter() - t0
    logger.info("Total training time: %.1f s (%.1f min)", elapsed, elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())