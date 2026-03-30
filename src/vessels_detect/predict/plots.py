"""
src/vessels_detect/predict/plots.py
-------------------------------------
Evaluation visualisations: per-class metric bar charts and confusion matrix.

All functions accept pre-computed data structures and write PNG files to
a caller-supplied directory.  No I/O is performed beyond writing the plot
files.

Dependencies: matplotlib (already required by Ultralytics).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_metrics_bar_chart(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    filename: str = "metrics_by_class.png",
) -> Path:
    """Save a grouped bar chart of Precision / Recall / F1 per class.

    The ``ALL_CLASSES`` micro-average row is excluded from the chart to
    keep the x-axis readable.

    Args:
        df:         Metrics DataFrame from
                    :func:`~src.vessels_detect.predict.metrics.build_metrics_dataframe`.
        output_dir: Directory where the PNG is written.
        filename:   Output filename.

    Returns:
        Resolved path to the written PNG.
    """
    import matplotlib.pyplot as plt

    plot_df = df[df["Class"] != "ALL_CLASSES"].set_index("Class")
    ax = plot_df[["Precision", "Recall", "F1"]].plot(
        kind="bar",
        figsize=(max(8, len(plot_df) * 1.5), 5),
        ylim=(0, 1.05),
        width=0.75,
        edgecolor="white",
    )
    ax.set_title("Per-class Precision / Recall / F1", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    out_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    logger.info("Bar chart saved → %s", out_path)
    return out_path.resolve()


def save_confusion_matrix(
    counts: Dict[int, Dict[str, int]],
    class_names: Dict[int, str],
    output_dir: Path,
    *,
    filename: str = "confusion_matrix.png",
) -> Path:
    """Save a per-class TP / FP / FN stacked bar (pseudo confusion matrix).

    A true N×N confusion matrix requires class-level match information;
    this function produces an equivalent per-class breakdown that is
    directly interpretable for object detection.

    Args:
        counts:      ``{class_id: {"TP": int, "FP": int, "FN": int}}``.
        class_names: ``{class_id: name}`` mapping.
        output_dir:  Directory where the PNG is written.
        filename:    Output filename.

    Returns:
        Resolved path to the written PNG.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [class_names.get(cid, str(cid)) for cid in sorted(counts)]
    tp_vals = [counts[cid]["TP"] for cid in sorted(counts)]
    fp_vals = [counts[cid]["FP"] for cid in sorted(counts)]
    fn_vals = [counts[cid]["FN"] for cid in sorted(counts)]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    ax.bar(x - width, tp_vals, width, label="TP", color="#4caf50")
    ax.bar(x,         fp_vals, width, label="FP", color="#f44336")
    ax.bar(x + width, fn_vals, width, label="FN", color="#ff9800")

    ax.set_title("TP / FP / FN by class", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    logger.info("Confusion matrix saved → %s", out_path)
    return out_path.resolve()
