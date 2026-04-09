"""
src/vessels_detect/predict/matcher.py
---------------------------------------
Greedy IoU matching between postprocessed predictions and ground-truth
annotations.

This module is the single source of truth for TP / FP / FN assignment
inside the evaluation pipeline.  It is deliberately free of I/O - it
operates purely on lists of :class:`~src.vessels_detect.postprocessing.spatial_filter.OBBBox`
objects and returns annotated copies.

Matching algorithm
~~~~~~~~~~~~~~~~~~
1.  Sort predictions by descending confidence.
2.  For each prediction, find the highest-IoU unmatched GT box of the
    **same class**.
3.  If ``IoU ≥ iou_threshold`` → mark prediction TP, consume the GT box.
4.  Predictions that find no match → FP.
5.  GT boxes that were never consumed → FN.

This is the standard PASCAL VOC greedy single-class matching procedure.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Tuple

from src.vessels_detect.postprocessing.spatial_filter import OBBBox


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match(
    predictions: List[OBBBox],
    ground_truths: List[OBBBox],
    iou_threshold: float,
) -> Tuple[List[OBBBox], List[OBBBox]]:
    """Assign TP / FP labels to predictions and FN labels to unmatched GT.

    Args:
        predictions:   Postprocessed prediction boxes (``OBBBox`` with
                       ``confidence`` set, ``label`` may be empty).
        ground_truths: Ground-truth boxes (``confidence = NaN``).
        iou_threshold: Minimum IoU for a prediction–GT pair to be a TP.

    Returns:
        ``(labelled_preds, labelled_gts)`` where every element has its
        ``label`` attribute set to ``"TP"``, ``"FP"``, or ``"FN"``.
    """
    # Sort predictions: highest confidence first.
    sorted_preds = sorted(
        enumerate(predictions),
        key=lambda t: t[1].confidence if not math.isnan(t[1].confidence) else 0.0,
        reverse=True,
    )

    labels_pred: List[str] = ["FP"] * len(predictions)
    labels_gt:   List[str] = ["FN"] * len(ground_truths)

    for pred_idx, pred in sorted_preds:
        best_iou  = iou_threshold   # must meet threshold to count as TP
        best_gi   = -1

        for gi, gt in enumerate(ground_truths):
            if labels_gt[gi] == "TP":
                continue                   # already consumed
            if gt.class_id != pred.class_id:
                continue                   # class mismatch

            iou = _polygon_iou(pred, gt)
            if iou >= best_iou:
                best_iou = iou
                best_gi  = gi

        if best_gi >= 0:
            labels_pred[pred_idx] = "TP"
            labels_gt[best_gi]    = "TP"   # mark GT as matched

    labelled_preds = [
        replace(p, label=labels_pred[i]) for i, p in enumerate(predictions)
    ]
    labelled_gts = [
        replace(g, label=labels_gt[i]) for i, g in enumerate(ground_truths)
    ]
    return labelled_preds, labelled_gts


def compute_per_class_counts(
    labelled_preds: List[OBBBox],
    labelled_gts: List[OBBBox],
) -> Dict[int, Dict[str, int]]:
    """Aggregate TP / FP / FN counts by class_id.

    Args:
        labelled_preds: Output of :func:`match` - predictions with labels.
        labelled_gts:   Output of :func:`match` - GT boxes with labels.

    Returns:
        ``{class_id: {"TP": int, "FP": int, "FN": int}}`` for every class
        that appears in either list.
    """
    counts: Dict[int, Dict[str, int]] = {}

    def _ensure(cid: int) -> None:
        if cid not in counts:
            counts[cid] = {"TP": 0, "FP": 0, "FN": 0}

    for pred in labelled_preds:
        _ensure(pred.class_id)
        if pred.label == "TP":
            counts[pred.class_id]["TP"] += 1
        else:
            counts[pred.class_id]["FP"] += 1

    for gt in labelled_gts:
        _ensure(gt.class_id)
        if gt.label == "FN":
            counts[gt.class_id]["FN"] += 1

    return counts


# ---------------------------------------------------------------------------
# Internal geometry helper
# ---------------------------------------------------------------------------

def _polygon_iou(a: OBBBox, b: OBBBox) -> float:
    """Compute polygon IoU between two :class:`OBBBox` instances.

    Args:
        a: First box.
        b: Second box.

    Returns:
        IoU in ``[0, 1]``.  Returns ``0.0`` on degenerate input.
    """
    try:
        inter = a.polygon.intersection(b.polygon).area
        union = a.polygon.union(b.polygon).area
        return inter / union if union > 0 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0
