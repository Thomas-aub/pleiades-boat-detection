"""
src/vessels_detect/postprocessing/steps/coastline.py
------------------------------------------------------
Postprocessing step that removes predicted polygons lying outside the
operational coastal range.

Algorithm
~~~~~~~~~
For each ``<stem>.geojson`` in ``predicted/``, this step looks for a
matching ``<stem>.geojson`` in ``coastlines/``.  Predictions whose
intersection with the coastline mask is below *min_area_fraction* are
dropped.  The filter is equivalent to :func:`keep_inside_buffer` from the
spatial filtering module, applied here in a file-pair loop.

Filter criterion
~~~~~~~~~~~~~~~~
::

    (prediction ∩ coastline_union).area / prediction.area  ≥  min_area_fraction

A prediction is **kept** only when this condition holds.

Output
~~~~~~
Filtered GeoJSON files are written to ``postprocessed/`` preserving the
original filename.  Files without a matching coastline mask are copied
unchanged with a warning.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

from shapely.geometry import MultiPolygon, shape

from src.vessels_detect.postprocessing.steps.base import BaseStep
from src.vessels_detect.utils.crs import load_mask_wgs84

logger = logging.getLogger(__name__)


class CoastlineFilter(BaseStep):
    """Remove predictions whose overlap with the coastline mask is insufficient."""

    NAME = "coastline_filter"

    def run(self, cfg: dict) -> int:
        """Apply coastline spatial filter to all predicted GeoJSON files.

        Args:
            cfg: Resolved config dict.  Consumed keys::

                cfg["paths"]["predicted"]        – input predictions dir
                cfg["paths"]["coastlines"]       – coastline masks dir
                cfg["paths"]["postprocessed"]    – output dir
                cfg["coastline_filter"]["min_area_fraction"]

        Returns:
            Total number of polygons removed across all files.
        """
        predicted_dir:    Path  = cfg["paths"]["predicted"]
        coastlines_dir:   Path  = cfg["paths"]["coastlines"]
        output_dir:       Path  = cfg["paths"]["postprocessed"]
        min_frac:         float = cfg["coastline_filter"]["min_area_fraction"]

        output_dir.mkdir(parents=True, exist_ok=True)

        pred_files = sorted(predicted_dir.glob("*.geojson"))
        if not pred_files:
            logger.warning("No .geojson files found in '%s'.", predicted_dir)
            return 0

        total_removed = 0

        for pred_path in pred_files:
            mask_path = coastlines_dir / pred_path.name
            out_path  = output_dir / pred_path.name

            if not mask_path.exists():
                logger.warning(
                    "No coastline mask for '%s' — copying unchanged.", pred_path.name
                )
                shutil.copy2(pred_path, out_path)
                continue

            coastline_mask = load_mask_wgs84(mask_path)
            kept, removed  = _filter_features(
                pred_path, coastline_mask, min_frac, criterion="area_inside"
            )
            _write_geojson(pred_path, kept, out_path)

            total_removed += removed
            logger.info(
                "CoastlineFilter '%s': kept %d, removed %d (min_area_fraction=%.2f).",
                pred_path.name, len(kept), removed, min_frac,
            )

        return total_removed


# ---------------------------------------------------------------------------
# Shared filter logic (also used by BuildingsFilter for consistency)
# ---------------------------------------------------------------------------

def _filter_features(
    pred_path: Path,
    mask: MultiPolygon,
    threshold: float,
    criterion: str,
) -> Tuple[List[dict], int]:
    """Filter GeoJSON features based on a spatial overlap criterion.

    Args:
        pred_path:  Path to the input ``.geojson`` predictions file.
        mask:       Shapely union mask to test against.
        threshold:  Numeric threshold for the chosen criterion.
        criterion:  ``"area_inside"``  — keep when
                    ``(feature ∩ mask).area / feature.area ≥ threshold``
                    (used by coastline filter).

                    ``"area_overlap"`` — drop when
                    ``(feature ∩ mask).area / feature.area ≥ threshold``
                    (used by buildings filter).

    Returns:
        ``(kept_features, n_removed)`` where ``kept_features`` is the
        filtered list of raw GeoJSON Feature dicts.
    """
    with open(pred_path, encoding="utf-8") as fh:
        collection = json.load(fh)

    features = collection.get("features", [])
    kept: List[dict] = []
    removed: int = 0

    for feat in features:
        geom_dict = feat.get("geometry")
        if geom_dict is None:
            removed += 1
            continue

        try:
            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-14:
                removed += 1
                continue

            frac = poly.intersection(mask).area / poly.area

            if criterion == "area_inside":
                passes = frac >= threshold
            elif criterion == "area_overlap":
                passes = frac < threshold
            else:
                raise ValueError(f"Unknown filter criterion: '{criterion}'")

            if passes:
                kept.append(feat)
            else:
                removed += 1

        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping malformed feature in '%s': %s", pred_path.name, exc)
            removed += 1

    return kept, removed


def _write_geojson(
    source_path: Path,
    features: List[dict],
    output_path: Path,
) -> None:
    """Write a filtered FeatureCollection to *output_path*.

    Preserves the top-level ``metadata`` field from the source collection if
    present, then overwrites it with updated counts.

    Args:
        source_path:  Original prediction file (for metadata inheritance).
        features:     Filtered list of GeoJSON Feature dicts to write.
        output_path:  Destination path; parent directories must already exist.
    """
    with open(source_path, encoding="utf-8") as fh:
        original = json.load(fh)

    collection: dict = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            **original.get("metadata", {}),
            "total_features": len(features),
        },
    }

    tmp = output_path.with_suffix(".tmp.geojson")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=2, ensure_ascii=False)
    tmp.replace(output_path)