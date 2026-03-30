"""
src/vessels_detect/predict/gt_loader.py
-----------------------------------------
Ground-truth GeoJSON loader for the evaluation pipeline.

Ground-truth files follow the same WGS-84 GeoJSON format as the raw
prediction files written by the predictor.  Each Feature must carry at
minimum a ``class_id`` property and a valid polygon geometry.

This module is the single place that knows how to deserialise GT files into
:class:`~src.vessels_detect.postprocessing.spatial_filter.OBBBox` objects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from shapely.geometry import shape

from src.vessels_detect.postprocessing.spatial_filter import OBBBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ground_truth(
    gt_path: Path,
    class_names: dict,
) -> List[OBBBox]:
    """Load a ground-truth GeoJSON file into a list of :class:`OBBBox`.

    Every Feature with a valid polygon geometry and a ``class_id`` property
    is converted to an :class:`OBBBox` with ``confidence = NaN`` (convention
    for GT annotations throughout this pipeline).

    Args:
        gt_path:     Path to the ``.geojson`` file.
        class_names: ``{class_id: name}`` mapping used to populate
                     ``OBBBox.class_name``.

    Returns:
        List of :class:`OBBBox` in WGS-84.  Features with missing or
        invalid geometry are skipped with a warning.

    Raises:
        FileNotFoundError: If *gt_path* does not exist.
    """
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    with open(gt_path, encoding="utf-8") as fh:
        collection = json.load(fh)

    boxes: List[OBBBox] = []

    for feat in collection.get("features", []):
        geom_dict = feat.get("geometry")
        props     = feat.get("properties", {})

        if geom_dict is None:
            logger.debug("Skipping GT feature with null geometry.")
            continue

        class_id = props.get("class_id")
        if class_id is None:
            logger.warning(
                "GT feature in '%s' has no class_id — skipping.", gt_path.name
            )
            continue

        try:
            poly = shape(geom_dict)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping malformed GT geometry in '%s': %s", gt_path.name, exc
            )
            continue

        class_id = int(class_id)
        boxes.append(OBBBox(
            polygon      = poly,
            class_id     = class_id,
            confidence   = float("nan"),
            source_image = gt_path.stem,
            class_name   = class_names.get(class_id, str(class_id)),
        ))

    logger.debug(
        "Loaded %d GT box(es) from '%s'.", len(boxes), gt_path.name
    )
    return boxes
