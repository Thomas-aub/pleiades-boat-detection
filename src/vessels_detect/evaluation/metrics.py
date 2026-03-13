"""
src/evaluation/metrics.py
--------------------------
Pure, testable metric functions extracted from ``model_results_384.ipynb``.

This module contains **no I/O, no Matplotlib, and no YOLO imports**.
Every function takes plain NumPy arrays or Python primitives and returns
scalars, arrays, or DataFrames.  This makes the logic straightforward to
unit-test and keeps the evaluation notebooks thin.

Contents
--------
Geometry
    :func:`shoelace_area` — polygon area via the shoelace formula.
    :func:`quad_sides`    — four side lengths of a quadrilateral.
    :func:`rotated_iou`   — rotated-polygon IoU via ``cv2.intersectConvexConvex``.

OBB statistics
    :func:`obb_geometric_stats`  — long-side, short-side, area, aspect ratio.
    :func:`build_size_dataframe` — aggregate per-class OBB size table.

Label parsing
    :func:`parse_label_file`   — parse one YOLO OBB ``.txt`` into (class, corners).
    :func:`count_split_labels` — per-class instance counts and tile statistics.

Per-class metrics
    :func:`compute_per_class_metrics` — P, R, F1, mAP50, mAP50-95, Support.
    :func:`compute_confusion_matrix`  — row-normalised confusion matrix.

Detection matching
    :func:`match_detections` — greedy IoU-based TP / FP / FN assignment.
    :func:`compute_crop_metrics` — precision / recall / F1 from TP/FP/FN counts.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# § 1  Geometry primitives
# =============================================================================

def shoelace_area(pts: Sequence[Tuple[float, float]]) -> float:
    """Compute polygon area using the shoelace (Gauss) formula.

    Args:
        pts: Ordered sequence of ``(x, y)`` vertex tuples.  The polygon
            does *not* need to be closed (first ≠ last).

    Returns:
        Unsigned area (same units as the coordinate system).

    Examples:
        >>> shoelace_area([(0,0),(1,0),(1,1),(0,1)])
        1.0
    """
    arr = np.asarray(pts, dtype=np.float64)
    x, y = arr[:, 0], arr[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def quad_sides(pts: Sequence[Tuple[float, float]]) -> List[float]:
    """Return the four Euclidean side lengths of a quadrilateral.

    Args:
        pts: Exactly 4 ``(x, y)`` tuples in order (consecutive vertices
            share an edge).

    Returns:
        List of 4 floats — side lengths in the order
        ``[|p0p1|, |p1p2|, |p2p3|, |p3p0|]``.
    """
    arr = np.asarray(pts, dtype=np.float64)
    return [
        float(np.hypot(arr[(i + 1) % 4, 0] - arr[i, 0],
                       arr[(i + 1) % 4, 1] - arr[i, 1]))
        for i in range(4)
    ]


def rotated_iou(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Compute IoU between two convex quadrilaterals (OBBs).

    Uses ``cv2.intersectConvexConvex`` to compute the intersection polygon,
    then derives union via the individual areas.

    Args:
        pts_a: Shape ``(4, 2)`` — corners of OBB A in pixel or CRS space.
        pts_b: Shape ``(4, 2)`` — corners of OBB B in the same space.

    Returns:
        IoU value in ``[0, 1]``.
    """
    a32 = pts_a.astype(np.float32)
    b32 = pts_b.astype(np.float32)

    ret, inter = cv2.intersectConvexConvex(a32, b32)
    if ret <= 0 or inter is None:
        return 0.0

    inter_pts  = inter.reshape(-1, 2)
    inter_area = shoelace_area(inter_pts)
    area_a     = shoelace_area(pts_a)
    area_b     = shoelace_area(pts_b)
    union_area  = area_a + area_b - inter_area
    return inter_area / (union_area + 1e-12)


# =============================================================================
# § 2  OBB geometric statistics
# =============================================================================

def obb_geometric_stats(
    pts_px: Sequence[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """Compute geometric statistics for a single OBB.

    Args:
        pts_px: Exactly 4 ``(x, y)`` pixel-space corner tuples.

    Returns:
        Tuple of ``(long_side, short_side, area, aspect_ratio)`` where:

        - ``long_side``   — length of the longest OBB edge (px).
        - ``short_side``  — length of the shortest OBB edge (px).
        - ``area``        — polygon area in px² (via shoelace formula).
        - ``aspect_ratio``— ``long_side / short_side``.
    """
    sides      = quad_sides(pts_px)
    long_side  = max(sides)
    short_side = min(sides)
    area       = shoelace_area(pts_px)
    aspect     = long_side / (short_side + 1e-9)
    return long_side, short_side, area, aspect


def build_size_dataframe(
    labels_root: Path,
    splits: Sequence[str],
    class_names: Dict[int, str],
    tile_size: int = 640,
) -> pd.DataFrame:
    """Build a per-class OBB size statistics DataFrame.

    Reads all YOLO OBB label files in the specified splits and computes
    mean / std of long-side, short-side, area, and aspect ratio for each
    class.

    Args:
        labels_root: Root directory containing ``<split>/`` subdirectories
            of ``.txt`` label files.
        splits: List of split names to include (e.g., ``["train", "val", "test"]``).
        class_names: Mapping ``{class_id: class_name}``.
        tile_size: Tile edge length in pixels.  Used to convert normalised
            YOLO coordinates to absolute pixel values.

    Returns:
        DataFrame with one row per class and columns:
        ``Class ID``, ``Class``, ``Count``,
        ``Long mean (px)``, ``Long std``,
        ``Short mean (px)``, ``Short std``,
        ``Area mean (px²)``, ``Aspect ratio``.
    """
    stats: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"long": [], "short": [], "area": []}
    )

    for split in splits:
        split_dir = labels_root / split
        if not split_dir.exists():
            logger.warning("build_size_dataframe: '%s' not found, skipped.", split_dir)
            continue

        for txt in split_dir.glob("*.txt"):
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts   = line.split()
                cid     = int(parts[0])
                coords  = list(map(float, parts[1:9]))
                pts_px  = [
                    (coords[2 * k] * tile_size, coords[2 * k + 1] * tile_size)
                    for k in range(4)
                ]
                long, short, area, _ = obb_geometric_stats(pts_px)
                stats[cid]["long"].append(long)
                stats[cid]["short"].append(short)
                stats[cid]["area"].append(area)

    rows = []
    for cid, s in sorted(stats.items()):
        if not s["long"]:
            continue
        long_arr  = np.array(s["long"])
        short_arr = np.array(s["short"])
        rows.append({
            "Class ID":       cid,
            "Class":          class_names.get(cid, str(cid)),
            "Count":          len(long_arr),
            "Long mean (px)": float(np.mean(long_arr)),
            "Long std":       float(np.std(long_arr)),
            "Short mean (px)": float(np.mean(short_arr)),
            "Short std":      float(np.std(short_arr)),
            "Area mean (px²)": float(np.mean(s["area"])),
            "Aspect ratio":   float(np.mean(long_arr / (short_arr + 1e-9))),
        })

    return pd.DataFrame(rows).round(2)


# =============================================================================
# § 3  Label file parsing
# =============================================================================

def parse_label_file(
    txt_path: Path,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, np.ndarray]]:
    """Parse a YOLO OBB ``.txt`` label file into (class_id, pixel_corners) pairs.

    Args:
        txt_path: Path to the YOLO OBB label file.
        img_w: Image width in pixels (used to denormalise x-coordinates).
        img_h: Image height in pixels (used to denormalise y-coordinates).

    Returns:
        List of ``(class_id, corners)`` where ``corners`` is a
        ``(4, 2)`` float32 array of pixel-space ``(col, row)`` pairs.
        Returns an empty list if the file is absent or empty.
    """
    if not txt_path.exists():
        return []

    boxes: List[Tuple[int, np.ndarray]] = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts  = line.split()
        cid    = int(parts[0])
        coords = list(map(float, parts[1:9]))
        pts    = np.array(
            [[coords[2 * k] * img_w, coords[2 * k + 1] * img_h] for k in range(4)],
            dtype=np.float32,
        )
        boxes.append((cid, pts))
    return boxes


def count_split_labels(
    labels_dir: Path,
) -> Tuple[Dict[int, int], int, int]:
    """Count per-class instances and tile statistics in a label directory.

    Args:
        labels_dir: Directory containing YOLO ``.txt`` label files.

    Returns:
        Tuple of:
            - ``counts``: ``{class_id: instance_count}`` across all files.
            - ``n_tiles``: Total number of ``.txt`` files.
            - ``n_annotated``: Number of non-empty ``.txt`` files.
    """
    counts: Dict[int, int] = defaultdict(int)
    n_annotated = 0
    txt_files   = list(labels_dir.glob("*.txt"))

    for txt in txt_files:
        lines = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
        if lines:
            n_annotated += 1
        for ln in lines:
            counts[int(ln.split()[0])] += 1

    return dict(counts), len(txt_files), n_annotated


def build_distribution_dataframe(
    labels_root: Path,
    splits: Sequence[str],
    class_names: Dict[int, str],
) -> pd.DataFrame:
    """Build a class-distribution table across splits.

    Args:
        labels_root: Root directory containing ``<split>/`` label directories.
        splits: Split names to include.
        class_names: ``{class_id: class_name}`` mapping.

    Returns:
        DataFrame with one row per class and one column per split plus
        a ``Total`` column.
    """
    split_data = {
        s: count_split_labels(labels_root / s)
        for s in splits
        if (labels_root / s).exists()
    }

    all_cids = sorted({cid for s_data in split_data.values() for cid in s_data[0]})
    rows = []
    for cid in all_cids:
        row: Dict[str, Any] = {"Class": class_names.get(cid, str(cid))}
        for split in splits:
            row[split.capitalize()] = split_data.get(split, ({}, 0, 0))[0].get(cid, 0)
        row["Total"] = sum(
            split_data.get(s, ({}, 0, 0))[0].get(cid, 0) for s in splits
        )
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# § 4  Per-class performance metrics
# =============================================================================

def compute_per_class_metrics(
    metrics_obj: Any,
    class_names: Dict[int, str],
    labels_root: Path,
    test_split: str = "test",
) -> pd.DataFrame:
    """Build the per-class P / R / F1 / mAP50 / mAP50-95 table.

    Args:
        metrics_obj: The ``ultralytics`` metrics object returned by
            ``model.val()``.  Must expose ``metrics_obj.box`` with
            attributes ``p``, ``r``, ``maps``, ``ap50``, ``mp``, ``mr``,
            ``map50``, ``map``.
        class_names: ``{class_id: class_name}`` mapping.
        labels_root: Root of the labels directory (for Support count).
        test_split: Split to count ground-truth support from.

    Returns:
        DataFrame with columns:
        ``Class``, ``Support``, ``Precision``, ``Recall``, ``F1-score``,
        ``mAP50``, ``mAP50-95``.
        A ``**Global**`` row is appended at the end.
    """
    # Ground-truth support counts.
    gt_counts: Counter[int] = Counter()
    test_dir = labels_root / test_split
    if test_dir.exists():
        for txt in test_dir.glob("*.txt"):
            for ln in txt.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    gt_counts[int(ln.split()[0])] += 1

    support = [gt_counts.get(i, 0) for i in range(len(class_names))]

    box = metrics_obj.box
    ap50_arr = box.ap50 if hasattr(box, "ap50") else [float("nan")] * len(class_names)

    rows = []
    for cid, name in class_names.items():
        p  = float(box.p[cid])
        r  = float(box.r[cid])
        f1 = 2 * p * r / (p + r + 1e-9)
        rows.append({
            "Class":     name,
            "Support":   support[cid],
            "Precision": p,
            "Recall":    r,
            "F1-score":  f1,
            "mAP50":     float(ap50_arr[cid]),
            "mAP50-95":  float(box.maps[cid]),
        })

    p_g, r_g = float(box.mp), float(box.mr)
    rows.append({
        "Class":     "**Global**",
        "Support":   sum(support),
        "Precision": p_g,
        "Recall":    r_g,
        "F1-score":  2 * p_g * r_g / (p_g + r_g + 1e-9),
        "mAP50":     float(box.map50),
        "mAP50-95":  float(box.map),
    })

    df = pd.DataFrame(rows)
    df = pd.concat([
        df[df["Class"] != "**Global**"].sort_values("mAP50-95", ascending=False),
        df[df["Class"] == "**Global**"],
    ]).reset_index(drop=True)

    return df.round(4)


def compute_confusion_matrix(
    metrics_obj: Any,
    class_names: Dict[int, str],
) -> Tuple[np.ndarray, List[str]]:
    """Extract and row-normalise the confusion matrix from ``model.val()``.

    Args:
        metrics_obj: The ``ultralytics`` metrics object.
        class_names: ``{class_id: class_name}`` mapping.

    Returns:
        Tuple of:
            - ``cm_norm``: Row-normalised confusion matrix as a ``(nc, nc)``
              float array (background row/column excluded).
            - ``labels``: Class name labels for axes.
    """
    cm_raw = metrics_obj.confusion_matrix.matrix  # (nc+1, nc+1)
    # Exclude the background row/column (last row/col).
    nc     = len(class_names)
    cm_raw = cm_raw[:nc, :nc]
    row_sums = cm_raw.sum(axis=1, keepdims=True) + 1e-9
    cm_norm  = cm_raw / row_sums
    labels   = [class_names[i] for i in range(nc)]
    return cm_norm, labels


# =============================================================================
# § 5  Detection matching (TP / FP / FN)
# =============================================================================

def match_detections(
    gt_boxes: List[Tuple[int, np.ndarray]],
    pred_boxes: List[Tuple[int, np.ndarray, float]],
    iou_threshold: float = 0.30,
) -> Tuple[List[bool], List[bool]]:
    """Greedy IoU-based matching of predictions to ground-truth boxes.

    Pairs are sorted by descending IoU and consumed greedily (each GT and
    each prediction is matched at most once).

    Args:
        gt_boxes: List of ``(class_id, corners_px)`` ground-truth boxes,
            where ``corners_px`` has shape ``(4, 2)``.
        pred_boxes: List of ``(class_id, corners_px, confidence)``
            predicted boxes.
        iou_threshold: Minimum rotated-IoU for a prediction to be
            considered a true positive.

    Returns:
        Tuple of:
            - ``matched_pred``: Boolean list, length ``len(pred_boxes)``.
              ``True`` if the prediction was matched to a GT box (TP).
            - ``matched_gt``: Boolean list, length ``len(gt_boxes)``.
              ``True`` if the GT box was matched by a prediction.
    """
    n_gt   = len(gt_boxes)
    n_pred = len(pred_boxes)

    matched_pred = [False] * n_pred
    matched_gt   = [False] * n_gt

    if n_gt == 0 or n_pred == 0:
        return matched_pred, matched_gt

    # Build IoU matrix.
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
    for gi, (gt_cid, gt_pts) in enumerate(gt_boxes):
        for pi, (pr_cid, pr_pts, _) in enumerate(pred_boxes):
            iou_matrix[gi, pi] = rotated_iou(gt_pts, pr_pts)

    # Greedy assignment: highest IoU pairs first.
    pairs = sorted(
        [
            (iou_matrix[gi, pi], gi, pi)
            for gi in range(n_gt)
            for pi in range(n_pred)
            if iou_matrix[gi, pi] >= iou_threshold
        ],
        reverse=True,
    )

    for _, gi, pi in pairs:
        if not matched_gt[gi] and not matched_pred[pi]:
            matched_gt[gi]   = True
            matched_pred[pi] = True

    return matched_pred, matched_gt


def compute_crop_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 from TP / FP / FN counts.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.

    Returns:
        Tuple of ``(precision, recall, f1)``.  All values are in ``[0, 1]``.
    """
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def build_crop_summary(
    summary_rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Build a per-class TP / FP / FN breakdown from a list of crop records.

    Args:
        summary_rows: List of dicts with at least ``"category"`` and
            ``"class"`` keys, as populated by the crop-export loop.

    Returns:
        DataFrame with columns ``category``, ``class``, ``count``.
    """
    if not summary_rows:
        return pd.DataFrame(columns=["category", "class", "count"])

    df = pd.DataFrame(summary_rows)
    return (
        df.groupby(["category", "class"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["category", "class"])
        .reset_index(drop=True)
    )
