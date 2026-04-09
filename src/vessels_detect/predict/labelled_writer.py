"""
src/vessels_detect/predict/labelled_writer.py
----------------------------------------------
Writes the final labelled evaluation GeoJSON that merges postprocessed
predictions, ground-truth annotations, and deleted predictions into a
single FeatureCollection.

Output schema per Feature
~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------+--------------------------------------------+--------------+
| Property            | Description                                | Type         |
+=====================+============================================+==============+
| ``label``           | ``"TP"`` | ``"FP"`` | ``"FN"``            | str          |
+---------------------+--------------------------------------------+--------------+
| ``class_id``        | YOLO class index of this box               | int          |
+---------------------+--------------------------------------------+--------------+
| ``class_name``      | Human-readable class name                  | str          |
+---------------------+--------------------------------------------+--------------+
| ``confidence``      | Model score; ``null`` for FN and deleted   | float | null |
+---------------------+--------------------------------------------+--------------+
| ``ground_truth``    | GT class_id this prediction matched, or    | int | null   |
|                     | ``null`` for FP / FN                       |              |
+---------------------+--------------------------------------------+--------------+
| ``deleted``         | ``true`` when this prediction was removed  | bool         |
|                     | during postprocessing                      |              |
+---------------------+--------------------------------------------+--------------+

Deleted predictions
~~~~~~~~~~~~~~~~~~~
A prediction is considered *deleted* when it appears in the raw prediction
GeoJSON (``predictions_dir``) but is absent from the postprocessed GeoJSON
(``postprocessed_dir``).  Deleted boxes are included in the output with
``"label": "FP"`` (they were never passed to the matcher) and
``"deleted": true``.

This allows downstream analysis to distinguish:
- True FP: predicted, survived postprocessing, no GT match.
- Deleted: predicted, but removed by a postprocessing filter.
- FN: present in GT, never matched.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

from shapely.geometry import mapping

from src.vessels_detect.postprocessing.spatial_filter import OBBBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_labelled_geojson(
    labelled_preds: List[OBBBox],
    labelled_gts: List[OBBBox],
    deleted_preds: List[OBBBox],
    output_path: Path,
    *,
    indent: int = 2,
) -> Path:
    """Serialise labelled predictions, GT, and deleted boxes to GeoJSON.

    Args:
        labelled_preds:  Predictions with ``label`` set (TP or FP) by the
                         matcher.  These survived postprocessing.
        labelled_gts:    GT boxes with ``label`` set (TP if matched, FN if
                         not).
        deleted_preds:   Predictions removed during postprocessing.
        output_path:     Destination ``.geojson`` file.
        indent:          JSON indentation level.

    Returns:
        Resolved path to the written file.
    """
    features: list = []

    # Surviving predictions (TP / FP).
    for pred in labelled_preds:
        gt_cid = _find_gt_class(pred, labelled_gts)
        features.append(_prediction_feature(pred, gt_class_id=gt_cid, deleted=False))

    # GT annotations (FN only - TPs are already represented by a prediction).
    for gt in labelled_gts:
        if gt.label == "FN":
            features.append(_gt_feature(gt))

    # Deleted predictions.
    for pred in deleted_preds:
        features.append(_prediction_feature(pred, gt_class_id=None, deleted=True))

    tp_count = sum(1 for p in labelled_preds if p.label == "TP")
    fp_count = sum(1 for p in labelled_preds if p.label == "FP")
    fn_count = sum(1 for g in labelled_gts   if g.label == "FN")

    collection = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_features": len(features),
            "TP":      tp_count,
            "FP":      fp_count,
            "FN":      fn_count,
            "deleted": len(deleted_preds),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp.geojson")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=indent, ensure_ascii=False)
    tmp.replace(output_path)

    logger.info(
        "Labelled GeoJSON → %s  (TP=%d  FP=%d  FN=%d  deleted=%d)",
        output_path, tp_count, fp_count, fn_count, len(deleted_preds),
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _prediction_feature(
    box: OBBBox,
    *,
    gt_class_id: Optional[int],
    deleted: bool,
) -> dict:
    conf = None if math.isnan(box.confidence) else round(box.confidence, 4)
    return {
        "type": "Feature",
        "geometry": mapping(box.polygon),
        "properties": {
            "label":        "FP" if deleted else box.label,
            "class_id":     box.class_id,
            "class_name":   box.class_name,
            "confidence":   conf,
            "ground_truth": gt_class_id,
            "deleted":      deleted,
        },
    }


def _gt_feature(gt: OBBBox) -> dict:
    return {
        "type": "Feature",
        "geometry": mapping(gt.polygon),
        "properties": {
            "label":        "FN",
            "class_id":     gt.class_id,
            "class_name":   gt.class_name,
            "confidence":   None,
            "ground_truth": gt.class_id,
            "deleted":      False,
        },
    }


def _find_gt_class(pred: OBBBox, gts: List[OBBBox]) -> Optional[int]:
    """Return the class_id of the GT box matched to *pred*, or ``None``.

    We identify the GT match as the TP GT box of the same class whose
    polygon has the highest IoU with *pred*.  This is a best-effort lookup
    for the labelled GeoJSON; the authoritative match is in :mod:`matcher`.
    """
    if pred.label != "TP":
        return None
    best_iou  = 0.0
    best_cid: Optional[int] = None
    for gt in gts:
        if gt.label != "TP" or gt.class_id != pred.class_id:
            continue
        try:
            inter = pred.polygon.intersection(gt.polygon).area
            union = pred.polygon.union(gt.polygon).area
            iou   = inter / union if union > 0 else 0.0
        except Exception:  # noqa: BLE001
            continue
        if iou > best_iou:
            best_iou = iou
            best_cid = gt.class_id
    return best_cid
