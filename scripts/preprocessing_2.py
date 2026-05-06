"""
scripts/preprocessing_2.py
---------------------------
CLI wrapper for the SAHI-optimised Global Preprocessing Pipeline — v2.

This version replaces the old Stage 4 (image-level split) + Stage 5 (tiling)
workflow with a unified **Stage 6 (tiled_split)** that:

1. Tiles every processed GeoTIFF into **non-overlapping** patches.
2. Splits the tile pool into ``train`` (80 %) and ``val`` (20 %), ensuring
   every source image contributes proportionally to both sets.
3. Shards the train set into 8 numbered sub-folders, again mixing tiles from
   all source images across every shard.
4. Writes a ``metadata.csv`` summarising source-image distribution, per-class
   object counts, and tile counts for every output folder.

Output layout
~~~~~~~~~~~~~
::

    data/<number>/
      train/
        1/  images/*.tif  labels/*.txt
        2/  …
        …
        8/  images/*.tif  labels/*.txt
      val/
        images/*.tif
        labels/*.txt
      metadata.csv

Pipeline stages
~~~~~~~~~~~~~~~
::

    1 radiometric - percentile stretch + gamma on the full GeoTIFF
    2 spatial     - Upsampling via windowed rasterio I/O
    3 annotations - GeoJSON OBB → YOLO OBB (global normalisation)
    6 tiled_split - non-overlapping tiling + train/val split + sharding
                    + metadata CSV

Usage
~~~~~
Run all enabled stages::

    PYTHONPATH=. python scripts/preprocessing_2.py

Run specific stages only::

    PYTHONPATH=. python scripts/preprocessing_2.py --stages radiometric spatial

Use a custom config::

    PYTHONPATH=. python scripts/preprocessing_2.py \\
        --config configs/preprocessing_2.yaml \\
        --stages annotations tiled_split

Adjust log verbosity::

    PYTHONPATH=. python scripts/preprocessing_2.py --log-level DEBUG
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
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
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
        description=(
            "SAHI-optimised Global Preprocessing Pipeline v2 — "
            "tile-first split with train sharding."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing_2.yaml"),
        metavar="PATH",
        help="Path to the YAML configuration file "
             "(default: configs/preprocessing_2.yaml).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        metavar="STAGE",
        default=None,
        help=(
            "Stage names to execute, overriding the 'enabled' flags in the "
            "config.  Valid names: radiometric, spatial, annotations, "
            "tiled_split.  Default: all enabled stages from the config."
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

    logger.info("preprocessing_2.py - SAHI Global Preprocessing Pipeline v2")
    logger.info("Config : %s", args.config.resolve())
    if args.stages:
        logger.info("Stage override : %s", args.stages)

    # Import after logging is configured so module-level loggers pick up the
    # correct level.
    from src.vessels_detect.preprocessing.manager_2 import PreprocessingManager

    manager = PreprocessingManager(config_path=args.config)
    return manager.run(stages_override=args.stages)


if __name__ == "__main__":
    sys.exit(main())