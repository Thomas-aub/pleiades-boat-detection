"""
src/vessels_detect/evaluation/postprocess_eval.py
--------------------------------------------------
Evaluation helpers that operate **after Global NMS**, in CRS (metric) space.

Why a separate module?
    The existing :mod:`~src.vessels_detect.evaluation.metrics` helpers
    work in pixel space, tile-by-tile.  Once Global NMS has been applied,
    detections are a flat list of :class:`~src.vessels_detect.inference.postprocess.GlobalDetection`
    objects in the tile's native CRS (e.g. UTM).  Matching them against
    GT requires projecting GT pixel corners to CRS first, then computing
    rotated-polygon IoU in that shared metric coordinate system.

Public API
----------
load_split_gt_crs
    Load every GT annotation for a split and project corners to CRS space.
    Returns ``(class_id, crs_corners, crs_string)`` triples so the caller
    can group by CRS before comparing with detections.

match_global_detections_crs
    Greedy per-class, per-CRS bipartite matching between detections and GT.

build_performance_dataframe
    Aggregate TP/FP/FN counts into a Precision / Recall / F1 / Bias table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyproj
import rasterio

from src.vessels_detect.evaluation.metrics import parse_label_file
from src.vessels_detect.inference.postprocess import (
    GlobalDetection,
    project_corners_to_crs,
    rotated_iou_crs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_crs(raw: str) -> str:
    """Normalise any CRS string to a stable ``EPSG:XXXX`` authority code.

    Using authority codes for comparison avoids false mismatches between
    equivalent CRS representations (e.g. WKT vs proj-string vs EPSG).

    Args:
        raw: Any CRS string accepted by :class:`pyproj.CRS`.

    Returns:
        ``"EPSG:<code>"`` if a numeric authority code can be resolved,
        otherwise the original string unchanged.
    """
    try:
        crs  = pyproj.CRS(raw)
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else raw
    except Exception:  # noqa: BLE001
        return raw


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# (class_id, crs_corners (4, 2), normalised_crs_string)
_GtCrs = Tuple[int, np.ndarray, str]


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_split_gt_crs(
    images_dir: Path,
    labels_dir: Path,
    splits: List[str],
) -> List[_GtCrs]:
    """Load all GT annotations for the requested splits, projected to CRS space.

    For each tile, pixel-space OBB corners from the YOLO label file are
    projected to the tile's native CRS using the Affine transform embedded
    in the GeoTIFF.  The normalised CRS string is stored alongside the
    corners so that :func:`match_global_detections_crs` can restrict
    comparisons to annotations that share the same coordinate zone — preventing
    cross-UTM-zone comparisons whose coordinates have completely different
    numerical ranges and would always yield IoU = 0.

    Args:
        images_dir: Root directory containing ``<split>/`` sub-directories
            of GeoTIFF tiles (e.g. ``data/processed/images``).
        labels_dir: Root directory containing ``<split>/`` sub-directories
            of YOLO ``.txt`` label files (e.g. ``data/processed/labels``).
        splits: List of split names to load (e.g. ``["test"]``).

    Returns:
        Flat list of ``(class_id, crs_corners, crs_string)`` triples.
        ``crs_corners`` has shape ``(4, 2)`` — ``(easting, northing)`` pairs
        in the tile's native CRS.  ``crs_string`` is an ``EPSG:XXXX``
        authority code when resolvable, otherwise the raw CRS string.
    """
    gt_list: List[_GtCrs] = []

    for split in splits:
        split_img_dir = images_dir / split
        split_lbl_dir = labels_dir / split

        if not split_img_dir.exists():
            logger.warning("load_split_gt_crs: images/%s/ not found — skipping.", split)
            continue

        img_paths = sorted(split_img_dir.glob("*.tif"))
        logger.info("Loading GT for split '%s' — %d tile(s).", split, len(img_paths))

        for img_path in img_paths:
            try:
                with rasterio.open(img_path) as ds:
                    tile_transform = ds.transform
                    tile_w, tile_h = ds.width, ds.height
                    crs_str        = _norm_crs(str(ds.crs))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cannot open '%s': %s — skipping.", img_path.name, exc)
                continue

            lbl_path = split_lbl_dir / (img_path.stem + ".txt")
            gt_pixel = parse_label_file(lbl_path, tile_w, tile_h)

            for class_id, pixel_corners in gt_pixel:
                crs_corners = project_corners_to_crs(pixel_corners, tile_transform)
                gt_list.append((class_id, crs_corners, crs_str))

    logger.info(
        "GT loaded — %d annotation(s) across %d split(s).", len(gt_list), len(splits)
    )
    return gt_list


# ---------------------------------------------------------------------------
# Global matching
# ---------------------------------------------------------------------------

def match_global_detections_crs(
    gt_list: List[_GtCrs],
    detections: List[GlobalDetection],
    iou_threshold: float,
) -> Tuple[List[bool], List[bool]]:
    """Greedy bipartite matching between CRS-space detections and GT.

    Matching is **per-class and per-CRS zone**: a detection can only match
    a GT annotation of the **same class** whose CRS normalises to the **same
    authority code**.  This is the critical guard against cross-UTM-zone
    comparisons which produce IoU = 0 for every pair (coordinates are
    numerically incompatible across zones even though both are valid UTM).

    Detections are processed in descending confidence order.  Each GT
    annotation can be matched at most once (greedy one-to-one assignment).

    Args:
        gt_list: Output of :func:`load_split_gt_crs` — list of
            ``(class_id, crs_corners, crs_string)`` triples.
        detections: Output of :class:`~src.vessels_detect.inference.postprocess.GlobalNMS`
            — list of :class:`GlobalDetection` objects, each carrying a
            ``.crs`` attribute.
        iou_threshold: Minimum rotated-polygon IoU (inclusive) to count
            a pair as a match.

    Returns:
        ``(m_pred, m_gt)`` where:
            - ``m_pred[i]`` is ``True`` if detection ``i`` matched a GT (→ TP).
            - ``m_gt[j]``   is ``True`` if GT ``j`` was matched (→ not FN).
    """
    m_pred: List[bool] = [False] * len(detections)
    m_gt:   List[bool] = [False] * len(gt_list)

    # Normalise detection CRS strings once up-front.
    det_crs_norm: List[str] = [_norm_crs(d.crs) for d in detections]

    # Process detections in descending confidence order.
    order = sorted(
        range(len(detections)),
        key=lambda i: detections[i].confidence,
        reverse=True,
    )

    for pi in order:
        det     = detections[pi]
        det_crs = det_crs_norm[pi]
        best_iou = iou_threshold   # must meet or beat threshold
        best_gi  = -1

        for gi, (gt_cid, gt_corners, gt_crs) in enumerate(gt_list):
            if m_gt[gi]:
                continue                        # already consumed by a higher-conf detection
            if gt_cid != det.class_id:
                continue                        # class mismatch
            if gt_crs != det_crs:
                continue                        # CRS zone mismatch — skip to avoid IoU=0

            iou = rotated_iou_crs(det.crs_corners, gt_corners)
            if iou >= best_iou:
                best_iou = iou
                best_gi  = gi

        if best_gi >= 0:
            m_pred[pi]    = True
            m_gt[best_gi] = True

    return m_pred, m_gt


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_performance_dataframe(
    stats: Dict[int, Dict[str, int]],
    class_names: Dict[int, str],
) -> "pd.DataFrame":  # noqa: F821
    """Convert per-class TP/FP/FN counts into a Precision/Recall/F1/Bias table.

    Args:
        stats: ``{class_id: {"TP": int, "FP": int, "FN": int}}``.
        class_names: ``{class_id: class_name}`` from the YOLO model.

    Returns:
        :class:`pandas.DataFrame` with one row per class plus a micro-average
        ``ALL_CLASSES_MICRO`` row.
        Columns: ``Class``, ``TP``, ``FP``, ``FN``, ``Precision``, ``Recall``,
        ``F1-Score``, ``Bias (%)``.
    """
    import pandas as pd

    rows = []
    for cid, s in stats.items():
        tp, fp, fn = s["TP"], s["FP"], s["FN"]
        precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1         = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        sum_pred = tp + fp
        sum_gt   = tp + fn
        bias     = ((sum_pred - sum_gt) / sum_gt * 100) if sum_gt > 0 else 0.0

        rows.append({
            "Class":     class_names.get(cid, str(cid)),
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 4),
            "Recall":    round(recall,    4),
            "F1-Score":  round(f1,        4),
            "Bias (%)":  round(bias,      2),
        })

    df = pd.DataFrame(rows)

    total_tp = sum(r["TP"] for r in rows)
    total_fp = sum(r["FP"] for r in rows)
    total_fn = sum(r["FN"] for r in rows)
    tot_p    = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    tot_r    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    tot_f1   = 2 * tot_p * tot_r / (tot_p + tot_r) if (tot_p + tot_r) > 0 else 0.0
    tot_bias = (
        ((total_tp + total_fp - total_tp - total_fn) / (total_tp + total_fn) * 100)
        if (total_tp + total_fn) > 0 else 0.0
    )

    df.loc[len(df)] = {
        "Class":     "ALL_CLASSES_MICRO",
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "Precision": round(tot_p,    4),
        "Recall":    round(tot_r,    4),
        "F1-Score":  round(tot_f1,   4),
        "Bias (%)":  round(tot_bias, 2),
    }

    return df