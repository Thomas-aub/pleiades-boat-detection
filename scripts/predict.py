#!/usr/bin/env python3
"""
scripts/predict.py
-------------------
Entry-point for the full geospatial inference pipeline.

Pipeline stages (executed in order):
    1. **Predict** — Run YOLO-OBB inference over every GeoTIFF tile in the
       configured ``paths.tiles_dir``.  Each tile's embedded CRS and Affine
       transform are read directly from the file (no metadata CSV needed).
    2. **Global NMS** — Project all raw pixel-space detections into the
       tile's native CRS, then run greedy per-class NMS to suppress
       cross-tile duplicates caused by the tile overlap.
    3. **GeoJSON Export** — Re-project the kept detections from the tile
       CRS to WGS 84 (EPSG:4326) and write a standard GeoJSON
       FeatureCollection file.

Usage::

    python scripts/predict.py
    python scripts/predict.py --config configs/inference.yaml
    python scripts/predict.py --tiles-dir data/processed/images/val \\
                              --output    predictions/val_run.geojson
    python scripts/predict.py --eval   # also run model.val() and print metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from src.utils.config import load_config, Config
from src.inference.predictor import YoloPredictor
from src.inference.postprocess import GlobalNMS
from src.inference.geospatial import GeoJSONExporter


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level_name: str) -> None:
    """Configure root logging.

    Args:
        level_name: Logging level name (e.g. ``"INFO"``).
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
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
        description="Run geospatial YOLO-OBB inference and export GeoJSON predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Path to the inference config YAML (default: configs/inference.yaml).",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=None,
        help="Override the tiles directory from the config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output GeoJSON path from the config.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Also run model.val() on the configured split and log formal metrics.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Optional formal evaluation
# ---------------------------------------------------------------------------

def _run_formal_eval(cfg: Config, predictor: YoloPredictor) -> None:
    """Run ``model.val()`` and log the per-class metric table.

    Args:
        cfg: The full inference config.
        predictor: An already-loaded :class:`~src.inference.predictor.YoloPredictor`.
    """
    from src.evaluation.metrics import compute_per_class_metrics

    eval_cfg      = cfg.evaluation
    dataset_yaml  = str(eval_cfg.get("dataset_yaml", "data/dataset.yaml"))
    split         = str(eval_cfg.get("split", "test"))
    final_conf    = float(eval_cfg.get("final_conf", 0.10))
    final_iou     = float(eval_cfg.get("final_iou", 0.30))
    labels_root   = Path(str(cfg.paths.get("tiles_dir", "data/processed/images/test"))).parent.parent.parent / "labels"

    logger.info("Running formal evaluation on '%s' split …", split)
    model   = predictor.model
    metrics = model.val(
        data=dataset_yaml,
        split=split,
        conf=final_conf,
        iou=final_iou,
        save_json=False,
        plots=False,
        verbose=False,
    )

    logger.info("=" * 60)
    logger.info("  Formal Evaluation Results  (split=%s)", split)
    logger.info("=" * 60)
    logger.info("  mAP50    : %.4f", metrics.box.map50)
    logger.info("  mAP50-95 : %.4f", metrics.box.map)
    logger.info("  Precision: %.4f", metrics.box.mp)
    logger.info("  Recall   : %.4f", metrics.box.mr)

    try:
        df = compute_per_class_metrics(
            metrics, predictor.class_names, labels_root, test_split=split
        )
        logger.info("\n%s", df.to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not build per-class table: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Execute the full inference pipeline.

    Args:
        argv: Optional argument list (for testing).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    logger.info("Boat Detection — Geospatial Inference Pipeline")
    logger.info("Config: %s", args.config.resolve())

    # ── Load config ───────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, Exception) as exc:
        logger.critical("Failed to load config: %s", exc)
        return 1

    # CLI overrides for the two most common flags.
    paths_dict = cfg["paths"].to_dict()
    if args.tiles_dir is not None:
        paths_dict["tiles_dir"] = str(args.tiles_dir)
    if args.output is not None:
        paths_dict["output_geojson"] = str(args.output)
    cfg = cfg.merge({"paths": paths_dict})

    tiles_dir      = Path(str(cfg.paths.tiles_dir))
    output_geojson = Path(str(cfg.paths.output_geojson))

    logger.info("  Tiles dir : %s", tiles_dir)
    logger.info("  Output    : %s", output_geojson)

    t_total = time.perf_counter()

    # ── Stage 1: YOLO inference ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1 — YOLO Inference")
    logger.info("=" * 60)
    try:
        predictor = YoloPredictor(cfg)
        raw_predictions = list(predictor.predict_directory(tiles_dir))
    except (FileNotFoundError, RuntimeError) as exc:
        logger.critical("Prediction failed: %s", exc)
        return 1
    except Exception as exc:
        logger.critical("Unexpected error during prediction: %s", exc, exc_info=True)
        return 1

    # ── Optional formal evaluation ────────────────────────────────────
    if args.eval:
        try:
            _run_formal_eval(cfg, predictor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Formal evaluation failed (non-fatal): %s", exc)

    # ── Stage 2: Global NMS ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 2 — Global NMS")
    logger.info("=" * 60)
    nms_iou = float(cfg.postprocess.get("global_nms_iou", 0.30))
    try:
        nms           = GlobalNMS(iou_threshold=nms_iou)
        kept_dets     = nms.run(raw_predictions)
    except Exception as exc:
        logger.critical("Global NMS failed: %s", exc, exc_info=True)
        return 1

    # ── Stage 3: GeoJSON export ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 3 — GeoJSON Export")
    logger.info("=" * 60)
    indent = cfg.export.get("indent", 2)

    try:
        exporter = GeoJSONExporter(class_names=predictor.class_names)
        out_path = exporter.export(kept_dets, output_geojson, indent=indent)
    except Exception as exc:
        logger.critical("GeoJSON export failed: %s", exc, exc_info=True)
        return 1

    elapsed = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info("Inference pipeline complete in %.1fs.", elapsed)
    logger.info("  Detections (after global NMS) : %d", len(kept_dets))
    logger.info("  GeoJSON output                : %s", out_path)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
