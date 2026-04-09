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
                    "No coastline mask for '%s' - copying unchanged.", pred_path.name
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
    mask_poly: MultiPolygon,
    threshold: float,
    criterion: str,
) -> Tuple[List[dict], int]:
    """
    Filter GeoJSON features based on their spatial intersection with a mask.

    Args:
        pred_path: Path to the input GeoJSON predictions file.
        mask_poly: The Shapely MultiPolygon/Polygon representing the geographic mask.
        threshold: The fraction threshold used for the filtering logic.
        criterion: The filtering logic to apply. Supported values:
                   - 'area_inside' : Keep feature if intersection fraction >= threshold.
                   - 'area_overlap': Keep feature if intersection fraction < threshold.

    Returns:
        A tuple containing:
            - A list of retained GeoJSON feature dictionaries.
            - An integer representing the number of dropped features.

    Raises:
        ValueError: If an unknown filtering criterion is provided.
        RuntimeError: If a fatal geometric computation error occurs.
    """
    with open(pred_path, encoding="utf-8") as fh:
        original = json.load(fh)


    kept: List[dict] = []
    removed = 0

    # 1. Global mask sanitization
    # Invalid geometries (e.g., self-intersecting polygons) are fixed via buffer(0).
    if not mask_poly.is_valid:
        mask_poly = mask_poly.buffer(0)

    # If the mask becomes empty after sanitization, gracefully skip filtering
    if mask_poly.is_empty:
        logger.warning(
            "Mask for '%s' is empty or invalid after sanitization. "
            "Bypassing filter and retaining all features.",
            pred_path.name,
        )
        return original.get("features", []), 0

    for feat in original.get("features", []):
        try:
            poly = shape(feat["geometry"])

            # 2. Feature geometry sanitization
            if not poly.is_valid:
                poly = poly.buffer(0)

            # 3. Safeguard against zero-area anomalies (e.g., in WGS-84 decimal degrees)
            feature_area = poly.area
            if feature_area < 1e-12:
                logger.debug(
                    "Feature area near zero in '%s'. Automatically keeping to prevent division by zero.", 
                    pred_path.name
                )
                kept.append(feat)
                continue

            # 4. Intersection computation
            intersection = poly.intersection(mask_poly)
            frac = intersection.area / feature_area

            # 5. Application of the filtering criterion
            if criterion == "area_inside":
                # Keep if the feature is sufficiently inside the mask (e.g., Coastlines/Operational Range)
                passes = frac >= threshold
            elif criterion == "area_overlap":
                # Keep if the feature does NOT excessively overlap the mask (e.g., Buildings/Clouds)
                passes = frac < threshold
            else:
                raise ValueError(f"Unknown filter criterion: '{criterion}'")

            if passes:
                kept.append(feat)
            else:
                removed += 1

        except Exception as exc:  # noqa: BLE001
            # Unmask hidden errors during geometric calculations
            raise RuntimeError(
                f"Fatal geometric computation error in '{pred_path.name}': {exc}"
            ) from exc

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