"""
src/vessels_detect/predict/evaluation.py
------------------------------------------
Evaluation stage: matches postprocessed predictions against ground truth,
computes per-class metrics, saves results, and optionally writes labelled
GeoJSON files.

This class is called by
:class:`~src.vessels_detect.manager.PredictManager` only when
``pipeline.mode: evaluation``.  It has no knowledge of model weights,
inference, or preprocessing - it reads ready-made GeoJSON files from disk.

Per-image workflow
~~~~~~~~~~~~~~~~~~
For each source image stem present in ``postprocessed_dir``:

1.  Load the postprocessed predictions.
2.  Load the corresponding raw predictions (to find deleted boxes).
3.  Load the ground-truth annotations.
4.  Match predictions against GT using
    :func:`~src.vessels_detect.predict.matcher.match`.
5.  Accumulate TP / FP / FN counts across all images.
6.  Optionally write a labelled GeoJSON for this image.

After all images:

7.  Build the metrics DataFrame via
    :func:`~src.vessels_detect.predict.metrics.build_metrics_dataframe`.
8.  Print the metrics table to the log.
9.  Save ``results.csv``.
10. Save per-class metric bar chart and confusion matrix PNG.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.vessels_detect.predict.gt_loader      import load_ground_truth
from src.vessels_detect.predict.pred_loader    import load_predictions, find_deleted_predictions
from src.vessels_detect.predict.matcher        import match, compute_per_class_counts
from src.vessels_detect.predict.metrics        import build_metrics_dataframe
from src.vessels_detect.predict.plots          import save_metrics_bar_chart, save_confusion_matrix
from src.vessels_detect.predict.labelled_writer import write_labelled_geojson

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates postprocessed predictions against ground-truth GeoJSON files.

    Instantiated and called by
    :class:`~src.vessels_detect.manager.PredictManager`.  Stateless across
    ``run()`` calls.
    """

    def run(self, cfg: dict) -> int:
        """Execute the evaluation stage.

        Args:
            cfg: Fully resolved config dict from
                 :func:`~src.vessels_detect.manager.load_config`.

        Returns:
            Exit code: ``0`` on success, ``1`` on fatal error.
        """
        try:
            return self._run(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.critical("Evaluator failed: %s", exc, exc_info=True)
            return 1

    def _run(self, cfg: dict) -> int:
        postprocessed_dir: Path  = cfg["paths"]["postprocessed_dir"]
        predictions_dir:   Path  = cfg["paths"]["predictions_dir"]
        gt_dir:            Path  = cfg["paths"]["ground_truth_dir"]
        results_dir:       Path  = cfg["paths"]["results_dir"]
        class_names:       Dict  = {int(k): v for k, v in cfg["model"]["class_names"].items()}
        eval_cfg:          dict  = cfg.get("evaluation", {})

        iou_threshold:    float = eval_cfg.get("iou_threshold", 0.50)
        generate_geojson: bool  = eval_cfg.get("generate_geojson", True)
        geojson_indent:   int   = eval_cfg.get("geojson_indent", 2)

        results_dir.mkdir(parents=True, exist_ok=True)

        post_files = sorted(postprocessed_dir.glob("*.geojson"))
        if not post_files:
            logger.error("No postprocessed GeoJSON files found in '%s'.", postprocessed_dir)
            return 1

        # Aggregate counts across all images.
        global_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

        for post_path in post_files:
            stem   = post_path.stem
            gt_path  = gt_dir / post_path.name
            raw_path = predictions_dir / post_path.name

            if not gt_path.exists():
                logger.warning("No GT file for '%s' - skipping.", stem)
                continue

            logger.info("Evaluating: %s", stem)

            # ── Load ───────────────────────────────────────────────────────
            post_preds = load_predictions(post_path,  class_names)
            gt_boxes   = load_ground_truth(gt_path,   class_names)

            # Identify predictions removed during postprocessing.
            deleted: list = []
            if raw_path.exists():
                raw_preds = load_predictions(raw_path, class_names)
                deleted   = find_deleted_predictions(raw_preds, post_preds)
            else:
                logger.debug("Raw prediction file not found for '%s'.", stem)

            # ── Match ──────────────────────────────────────────────────────
            labelled_preds, labelled_gts = match(
                post_preds, gt_boxes, iou_threshold
            )

            # ── Accumulate counts ──────────────────────────────────────────
            image_counts = compute_per_class_counts(labelled_preds, labelled_gts)
            for cid, s in image_counts.items():
                for key in ("TP", "FP", "FN"):
                    global_counts[cid][key] += s[key]

            # ── Per-image labelled GeoJSON ─────────────────────────────────
            if generate_geojson:
                out_path = results_dir / f"{stem}_eval.geojson"
                write_labelled_geojson(
                    labelled_preds,
                    labelled_gts,
                    deleted,
                    out_path,
                    indent=geojson_indent,
                )

        if not global_counts:
            logger.warning("No evaluation data collected - check GT directory.")
            return 0

        # ── Global metrics ─────────────────────────────────────────────────
        df = build_metrics_dataframe(dict(global_counts), class_names)
        self._log_metrics(df)

        csv_path = results_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Results saved → %s", csv_path)

        save_metrics_bar_chart(df, results_dir)
        save_confusion_matrix(dict(global_counts), class_names, results_dir)

        return 0

    @staticmethod
    def _log_metrics(df) -> None:
        logger.info("")
        logger.info("Evaluation results:")
        logger.info("%s", df.to_string(index=False))
