#!/usr/bin/env python3
"""
scripts/train.py
-----------------
Entry-point for YOLO-OBB boat-detection model training.

Reads ``configs/train.yaml``, instantiates :class:`~src.models.yolo_trainer.YoloTrainer`,
and starts the training run.  All hyperparameters are driven by the YAML;
no values are hardcoded here.

Usage::

    python scripts/train.py
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --log-level DEBUG

    # Override a single key without editing the YAML (dot-separated path):
    python scripts/train.py --set training.epochs=50 training.batch_size=4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from src.utils.config import load_config
from src.models.yolo_trainer import YoloTrainer


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
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Keep ultralytics output at WARNING to avoid per-batch spam.
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
            "Override config values without editing the YAML. "
            "Use dot-separated paths, e.g. --set training.epochs=100."
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

    Args:
        raw: List of ``key=value`` strings where the key may contain dots
            to indicate nesting (e.g. ``"training.epochs=50"``).

    Returns:
        A nested dictionary suitable for :meth:`~src.utils.config.Config.merge`.
    """
    overrides: dict = {}
    for item in raw:
        if "=" not in item:
            logger.warning("Ignoring malformed override '%s' (missing '=').", item)
            continue
        key_path, _, raw_val = item.partition("=")
        # Attempt numeric coercion; fall back to string.
        try:
            value = int(raw_val)
        except ValueError:
            try:
                value = float(raw_val)
            except ValueError:
                value = raw_val  # type: ignore[assignment]

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
    """Train the YOLO-OBB model.

    Args:
        argv: Optional argument list (for testing).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    logger.info("Boat Detection — YOLO-OBB Training Script")
    logger.info("Config: %s", args.config.resolve())

    # ── Load & optionally patch the config ────────────────────────────
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, Exception) as exc:
        logger.critical("Failed to load config: %s", exc)
        return 1

    if args.set:
        overrides = _parse_overrides(args.set)
        cfg = cfg.merge(overrides)
        logger.info("Applied CLI overrides: %s", overrides)

    # ── Train ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        trainer = YoloTrainer(cfg)
        results = trainer.train()
    except FileNotFoundError as exc:
        logger.critical("File not found: %s", exc)
        return 1
    except Exception as exc:
        logger.critical("Training failed: %s", exc, exc_info=True)
        return 1

    elapsed = time.perf_counter() - t0
    logger.info("Total training time: %.1f s (%.1f min)", elapsed, elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
