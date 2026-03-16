"""
scripts/prep_data.py
---------------------
Entry-point for the Pleiades Neo boat-detection preprocessing pipeline.

Reads ``configs/data_prep.yaml`` and orchestrates the five pipeline stages
in the exact order required:

    Stage 1 — Tiling
        Slides a window over every raw GeoTIFF and saves each patch as a
        self-contained georeferenced GeoTIFF tile (CRS + Affine embedded).
        Output: ``<tiles_dir>/``

    Stage 2 — Annotation Conversion
        Converts GeoJSON OBB annotations into YOLO OBB ``.txt`` label files
        by projecting geo-coordinates into each tile's pixel space using the
        tile's embedded Affine transform.
        Output: ``<labels_raw_dir>/``

    Stage 3 — Train / Val / Test Split
        Groups tiles at the *source-image level* (zero spatial leakage) and
        moves them into ``<images_dir>/{train,val,test}/``.  Corresponding
        labels are moved into ``<labels_dir>/{train,val,test}/``.

    Stage 4 — Background Balancing  (train + val only)
        Moves excess background tiles to ``<archive_dir>/`` until the
        background fraction of each processed split matches
        ``balance.background_ratio``.

    Stage 5 — Upsampling  (train + val only)
        Resizes all tiles in the processed splits from ``tile_size`` to
        ``target_size`` using a high-quality interpolation filter.  The
        GeoTIFF Affine transform is updated to reflect the new pixel
        resolution; labels are unchanged (coordinates are scale-invariant).

Usage::

    python scripts/prep_data.py [--config configs/data_prep.yaml]
                                [--stages 1 2 3 4 5]
                                [--log-level INFO]


Example — run only stages 3 and 4::

    python scripts/prep_data.py --stages 3 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level_name: str) -> None:
    """Configure root-level logging with a human-readable format.

    Args:
        level_name: One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
            ``"ERROR"``, ``"CRITICAL"`` (case-insensitive).
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
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """Load and minimally validate the YAML configuration file.

    Args:
        config_path: Path to ``data_prep.yaml``.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        KeyError: If a required top-level section is absent.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    required_sections = ("paths", "tiling", "annotations", "split", "balance", "transforms")
    for section in required_sections:
        if section not in cfg:
            raise KeyError(
                f"Required config section '{section}' is missing from '{config_path}'."
            )

    return cfg


def _resolve_paths(cfg: dict) -> dict:
    """Convert all path strings in ``cfg["paths"]`` to absolute :class:`~pathlib.Path` objects.

    Relative paths are resolved relative to the current working directory,
    which is expected to be the project root when the script is run as::

        python scripts/prep_data.py

    Args:
        cfg: Full configuration dictionary (mutated in-place).

    Returns:
        The ``paths`` sub-dictionary with :class:`~pathlib.Path` values.
    """
    paths = {}
    for key, value in cfg["paths"].items():
        paths[key] = Path(value).resolve()
    cfg["paths"] = paths
    return paths


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _stage_1_tiling(cfg: dict) -> None:
    """Run Stage 1: tile raw GeoTIFFs → georeferenced GeoTIFF tiles.

    Args:
        cfg: Full configuration dictionary.
    """
    from src.vessels_detect.data.tiler import ImageTiler, TilerConfig

    logger.info("=" * 70)
    logger.info("STAGE 1 — Tiling")
    logger.info("=" * 70)

    paths  = cfg["paths"]
    config = TilerConfig.from_dict(cfg["tiling"])
    tiler  = ImageTiler(config)

    raw_dir   = paths["raw_dir"]
    tiles_dir = paths["tiles_dir"]

    tiles_dir.mkdir(parents=True, exist_ok=True)

    tile_paths = tiler.tile_directory(raw_dir=raw_dir, output_dir=tiles_dir)
    logger.info("Stage 1 complete — %d tile(s) written to '%s'.", len(tile_paths), tiles_dir)


def _stage_2_annotations(cfg: dict) -> None:
    """Run Stage 2: convert GeoJSON OBBs → YOLO OBB ``.txt`` files.

    Args:
        cfg: Full configuration dictionary.
    """
    from src.vessels_detect.data.annotations import AnnotationConverter, AnnotationConfig

    logger.info("=" * 70)
    logger.info("STAGE 2 — Annotation Conversion")
    logger.info("=" * 70)

    paths  = cfg["paths"]
    config = AnnotationConfig.from_dict(cfg["annotations"])
    conv   = AnnotationConverter(config)

    conv.convert_directory(
        tiles_dir=paths["tiles_dir"],
        raw_dir=paths["raw_dir"],
        labels_dir=paths["labels_raw_dir"],
    )
    logger.info(
        "Stage 2 complete — labels written to '%s'.", paths["labels_raw_dir"]
    )


def _stage_3_split(cfg: dict) -> None:
    """Run Stage 3: distribute tiles into train / val / test splits.

    Args:
        cfg: Full configuration dictionary.
    """
    from src.vessels_detect.data.split import DatasetSplitter, SplitConfig

    logger.info("=" * 70)
    logger.info("STAGE 3 — Train / Val / Test Split")
    logger.info("=" * 70)

    paths    = cfg["paths"]
    config   = SplitConfig.from_dict(cfg["split"])
    splitter = DatasetSplitter(config)

    result = splitter.split(
        tiles_dir=paths["tiles_dir"],
        labels_raw_dir=paths["labels_raw_dir"],
        images_dir=paths["images_dir"],
        labels_dir=paths["labels_dir"],
    )

    for split, tile_list in result.items():
        logger.info("  %-5s: %d tile(s)", split, len(tile_list))

    logger.info(
        "Stage 3 complete — tiles distributed under '%s'.", paths["images_dir"]
    )


def _stage_4_balance(cfg: dict) -> None:
    """Run Stage 4: balance background tiles in train + val splits.

    Args:
        cfg: Full configuration dictionary.
    """
    from src.vessels_detect.data.split import BackgroundBalancer, BalanceConfig

    logger.info("=" * 70)
    logger.info("STAGE 4 — Background Balancing")
    logger.info("=" * 70)

    paths    = cfg["paths"]
    bal_cfg  = cfg["balance"]
    config   = BalanceConfig.from_dict(bal_cfg)
    balancer = BackgroundBalancer(config)
    splits   = bal_cfg.get("apply_to_splits", ["train", "val"])

    balancer.balance(
        images_dir=paths["images_dir"],
        labels_dir=paths["labels_dir"],
        archive_dir=paths["archive_dir"],
        splits=splits,
    )

    logger.info(
        "Stage 4 complete — excess background moved to '%s'.", paths["archive_dir"]
    )


def _stage_5_upsample(cfg: dict) -> None:
    """Run Stage 5: upsample tiles in train + val splits.

    Args:
        cfg: Full configuration dictionary.
    """
    from src.vessels_detect.data.transforms import ImageUpsampler, UpsampleConfig

    logger.info("=" * 70)
    logger.info("STAGE 5 — Upsampling")
    logger.info("=" * 70)

    paths    = cfg["paths"]
    tr_cfg   = cfg["transforms"]
    config   = UpsampleConfig.from_dict(tr_cfg)
    upsampler = ImageUpsampler(config)
    splits   = tr_cfg.get("apply_to_splits", ["train", "val"])

    upsampler.upsample_splits(
        images_dir=paths["images_dir"],
        splits=splits,
    )

    logger.info("Stage 5 complete.")


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_FN = {
    1: _stage_1_tiling,
    2: _stage_2_annotations,
    3: _stage_3_split,
    4: _stage_4_balance,
    5: _stage_5_upsample,
}

_STAGE_NAMES = {
    1: "Tiling",
    2: "Annotation Conversion",
    3: "Train/Val/Test Split",
    4: "Background Balancing",
    5: "Upsampling",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description="Pleiades Neo boat-detection preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_prep.yaml"),
        help="Path to the YAML configuration file "
             "(default: configs/data_prep.yaml).",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=list(_STAGE_FN.keys()),
        default=list(_STAGE_FN.keys()),
        metavar="N",
        help="Stages to execute (1–5). Default: all stages.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Orchestrate the preprocessing pipeline.

    Args:
        argv: Optional argument list for testing. Uses ``sys.argv`` if
            ``None``.

    Returns:
        Exit code: ``0`` on success, ``1`` on fatal error.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    logger.info("Pleiades Neo — Boat Detection Preprocessing Pipeline")
    logger.info("Config : %s", args.config.resolve())
    logger.info("Stages : %s", args.stages)

    # ------------------------------------------------------------------
    # Load configuration.
    # ------------------------------------------------------------------
    try:
        cfg = _load_config(args.config)
    except (FileNotFoundError, KeyError, Exception) as exc:
        logger.critical("Failed to load configuration: %s", exc)
        return 1

    _resolve_paths(cfg)

    logger.info("Paths resolved:")
    for key, val in cfg["paths"].items():
        logger.info("  %-16s: %s", key, val)

    # ------------------------------------------------------------------
    # Execute requested stages in order.
    # ------------------------------------------------------------------
    pipeline_start = time.perf_counter()
    requested      = sorted(args.stages)

    for stage_id in requested:
        stage_fn   = _STAGE_FN[stage_id]
        stage_name = _STAGE_NAMES[stage_id]
        t0 = time.perf_counter()

        logger.info("")
        try:
            stage_fn(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.critical(
                "Stage %d (%s) failed with an unhandled exception: %s",
                stage_id, stage_name, exc,
                exc_info=True,
            )
            return 1

        elapsed = time.perf_counter() - t0
        logger.info(
            "Stage %d (%s) finished in %.1fs.", stage_id, stage_name, elapsed
        )

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "Pipeline complete.  Stages run: %s  |  Total time: %.1fs",
        requested, total_elapsed,
    )
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
