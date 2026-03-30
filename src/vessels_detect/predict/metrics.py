"""
src/vessels_detect/predict/metrics.py
---------------------------------------
Per-class metric computation, CSV serialisation, and console reporting.

This module is purely computational — it takes pre-computed TP / FP / FN
counts and returns a structured :class:`~pandas.DataFrame`.  All I/O
(loading GeoJSONs, writing CSV) is handled by the caller
(:class:`~src.vessels_detect.predict.evaluation.Evaluator`).

Metrics produced
~~~~~~~~~~~~~~~~
For each class and for a micro-average ``ALL_CLASSES`` row:

+-------------+---------------------------------------------------+
| Column      | Definition                                        |
+=============+===================================================+
| Precision   | TP / (TP + FP)                                    |
+-------------+---------------------------------------------------+
| Recall      | TP / (TP + FN)                                    |
+-------------+---------------------------------------------------+
| F1          | 2 · P · R / (P + R)                               |
+-------------+---------------------------------------------------+
| Bias (%)    | (FP − FN) / (TP + FN) × 100                       |
|             | > 0 → over-detection, < 0 → under-detection       |
+-------------+---------------------------------------------------+
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_metrics_dataframe(
    counts: Dict[int, Dict[str, int]],
    class_names: Dict[int, str],
) -> pd.DataFrame:
    """Convert per-class TP / FP / FN counts to a metrics DataFrame.

    Args:
        counts:      ``{class_id: {"TP": int, "FP": int, "FN": int}}``.
        class_names: ``{class_id: human_readable_name}`` mapping.

    Returns:
        :class:`~pandas.DataFrame` with one row per class plus a
        ``ALL_CLASSES`` micro-average row.  Columns: ``Class``, ``TP``,
        ``FP``, ``FN``, ``Precision``, ``Recall``, ``F1``, ``Bias (%)``.
    """
    rows = []
    for cid, s in sorted(counts.items()):
        rows.append(_class_row(cid, s, class_names.get(cid, str(cid))))

    df = pd.DataFrame(rows)

    # Micro-average across all classes.
    total_tp = sum(r["TP"] for r in rows)
    total_fp = sum(r["FP"] for r in rows)
    total_fn = sum(r["FN"] for r in rows)
    df.loc[len(df)] = _metrics_row(
        "ALL_CLASSES", total_tp, total_fp, total_fn
    )

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _class_row(
    class_id: int,
    counts: Dict[str, int],
    class_name: str,
) -> dict:
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    return _metrics_row(class_name, tp, fp, fn)


def _metrics_row(name: str, tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    bias = (fp - fn) / (tp + fn) * 100 if (tp + fn) > 0 else 0.0

    return {
        "Class":      name,
        "TP":         tp,
        "FP":         fp,
        "FN":         fn,
        "Precision":  round(precision, 4),
        "Recall":     round(recall,    4),
        "F1":         round(f1,        4),
        "Bias (%)":   round(bias,      2),
    }
