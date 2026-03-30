"""
scripts/preprocessing.py
-----------------------------
Thin CLI wrapper for the SAHI-optimised Global Preprocessing Pipeline.

This script is the sole entry-point for human operators and CI jobs.  All
orchestration logic lives in :mod:`src.vessels_detect.preprocessing.manager`;
this file only handles argument parsing, logging setup, and the process
exit code.

Pipeline stages
~~~~~~~~~~~~~~~
::

    1 radiometric — percentile stretch + gamma on the full GeoTIFF
    2 spatial     — Lanczos-4 upsampling via windowed rasterio I/O
    3 annotations — GeoJSON OBB → YOLO OBB (global normalisation)
    4 split       — image-level train / val / test (zero spatial leakage)
    5 tiling      — raw GeoTIFF tiling + YOLO OBB label projection
                    

Usage
~~~~~
Run all enabled stages::

    PYTHONPATH=. python scripts/preprocessing.py

Run specific stages only::

    PYTHONPATH=. python scripts/preprocessing.py --stages radiometric spatial

Use a custom config::

    PYTHONPATH=. python scripts/preprocessing.py \\
        --config configs/my_experiment.yaml \\
        --stages annotations split tiling

Adjust log verbosity::

    PYTHONPATH=. python scripts/preprocessing.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level_name: str) -> None:
    """Configure root-level logging with a consistent, readable format.

    Args:
        level_name: One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
            ``CRITICAL`` (case-insensitive).
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="SAHI-optimised Global Preprocessing Pipeline for vessel detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing.yaml"),
        metavar="PATH",
        help="Path to the YAML configuration file "
             "(default: configs/preprocessing.yaml).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        default=None,
        help=(
            "Stage names to execute, overriding the 'enabled' flags in the "
            "config.  Valid names: radiometric, spatial, annotations, split, "
            "tiling.  Default: all enabled stages from the config."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO).",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments and delegate to :class:`PreprocessingManager`.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: ``0`` on success, ``1`` on error.
    """
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)

    logger.info("preprocessing.py — SAHI Global Preprocessing Pipeline")
    logger.info("Config : %s", args.config.resolve())
    if args.stages:
        logger.info("Stage override : %s", args.stages)

    # Import here so logging is configured before any module-level loggers fire.
    from src.vessels_detect.preprocessing.manager import PreprocessingManager

    manager = PreprocessingManager(config_path=args.config)
    return manager.run(stages_override=args.stages)


if __name__ == "__main__":
    sys.exit(main())