"""
src/vessels_detect/predict/pred_loader.py
-------------------------------------------
Prediction GeoJSON loader for the evaluation pipeline.

Reads the GeoJSON files written by
:func:`~src.vessels_detect.postprocessing.geojson_writer.write_prediction_geojson`
back into :class:`~src.vessels_detect.postprocessing.spatial_filter.OBBBox`
objects.  Used both for the raw predictions (to identify deleted boxes) and
the postprocessed predictions (to build the final evaluation inputs).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List

from shapely.geometry import shape

from src.vessels_detect.postprocessing.spatial_filter import OBBBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_predictions(
    geojson_path: Path,
    class_names: dict,
) -> List[OBBBox]:
    """Load a prediction GeoJSON file into a list of :class:`OBBBox`.

    Args:
        geojson_path: Path to the ``.geojson`` file.
        class_names:  ``{class_id: name}`` mapping.

    Returns:
        List of :class:`OBBBox` in WGS-84.  Features with missing geometry
        or ``class_id`` are skipped with a debug log.

    Raises:
        FileNotFoundError: If *geojson_path* does not exist.
    """
    if not geojson_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {geojson_path}")

    with open(geojson_path, encoding="utf-8") as fh:
        collection = json.load(fh)

    boxes: List[OBBBox] = []

    for feat in collection.get("features", []):
        geom_dict = feat.get("geometry")
        props     = feat.get("properties", {})

        if geom_dict is None:
            continue

        class_id   = props.get("class_id")
        confidence = props.get("confidence")

        if class_id is None:
            logger.debug("Prediction feature in '%s' has no class_id.", geojson_path.name)
            continue

        try:
            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping malformed prediction geometry: %s", exc)
            continue

        class_id   = int(class_id)
        confidence = float(confidence) if confidence is not None else float("nan")

        boxes.append(OBBBox(
            polygon      = poly,
            class_id     = class_id,
            confidence   = confidence,
            source_image = geojson_path.stem,
            class_name   = class_names.get(class_id, str(class_id)),
        ))

    logger.debug(
        "Loaded %d prediction(s) from '%s'.", len(boxes), geojson_path.name
    )
    return boxes


def find_deleted_predictions(
    raw_predictions: List[OBBBox],
    postprocessed_predictions: List[OBBBox],
) -> List[OBBBox]:
    """Return predictions present in *raw* but absent from *postprocessed*.

    Deletion is detected by polygon centroid proximity (WGS-84) and class_id
    match, which avoids floating-point geometry equality issues while being
    robust for this use case (centroids of distinct vessels are always
    spatially separated).

    Args:
        raw_predictions:          All boxes from the raw prediction GeoJSON.
        postprocessed_predictions: Boxes that survived postprocessing.

    Returns:
        Subset of *raw_predictions* not present in *postprocessed_predictions*.
    """
    # Build a centroid set for fast lookup: (class_id, rounded_lon, rounded_lat)
    PRECISION = 8  # decimal degrees ~ 1 mm precision

    def _key(box: OBBBox):
        c = box.polygon.centroid
        return (box.class_id, round(c.x, PRECISION), round(c.y, PRECISION))

    survived_keys = {_key(b) for b in postprocessed_predictions}
    return [b for b in raw_predictions if _key(b) not in survived_keys]
