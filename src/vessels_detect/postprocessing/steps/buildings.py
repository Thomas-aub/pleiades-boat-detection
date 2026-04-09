"""
src/vessels_detect/postprocessing/steps/buildings.py
------------------------------------------------------
Postprocessing step that removes predicted polygons with excessive overlap
with building footprints.

Algorithm
~~~~~~~~~
For each ``<stem>.geojson`` in the *current working directory* (either
``predicted/`` on the first step, or ``postprocessed/`` when chained after
the coastline filter), this step looks for a matching ``<stem>.geojson`` in
``buildings/``.  Predictions whose intersection-over-prediction (IoP) with
any building footprint meets or exceeds *max_overlap_fraction* are dropped.

Filter criterion (IoP - intersection over prediction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    (prediction ∩ buildings_union).area / prediction.area  ≥  max_overlap_fraction

A prediction is **dropped** when this condition holds.  IoP is preferred
over IoU because buildings are typically much larger than vessels; IoU
would almost never trigger.

Chaining
~~~~~~~~
When the :class:`CoastlineFilter` runs first, its output lands in
``postprocessed/``.  :class:`BuildingsFilter` then reads from
``postprocessed/`` (``read_from_postprocessed: true`` in the YAML) so that
the two filters compose correctly.  When it runs standalone, it reads
directly from ``predicted/``.

Output
~~~~~~
Filtered GeoJSON files overwrite the files in ``postprocessed/``.  Files
without a matching buildings mask are left unchanged with a warning.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from src.vessels_detect.postprocessing.steps.base import BaseStep
from src.vessels_detect.postprocessing.steps.coastline import _filter_features, _write_geojson
from src.vessels_detect.utils.crs import load_mask_wgs84

logger = logging.getLogger(__name__)


class BuildingsFilter(BaseStep):
    """Remove predictions with excessive overlap with building footprints."""

    NAME = "buildings_filter"

    def run(self, cfg: dict) -> int:
        """Apply buildings exclusion filter to all GeoJSON files.

        Args:
            cfg: Resolved config dict.  Consumed keys::

                cfg["paths"]["predicted"]           – raw predictions dir
                cfg["paths"]["buildings"]           – building masks dir
                cfg["paths"]["postprocessed"]       – output (and possibly input) dir
                cfg["buildings_filter"]["max_overlap_fraction"]
                cfg["buildings_filter"]["read_from_postprocessed"]

        Returns:
            Total number of polygons removed across all files.
        """
        predicted_dir:    Path  = cfg["paths"]["predicted"]
        buildings_dir:    Path  = cfg["paths"]["buildings"]
        output_dir:       Path  = cfg["paths"]["postprocessed"]
        max_frac:         float = cfg["buildings_filter"]["max_overlap_fraction"]
        # When chained after CoastlineFilter, read the already-filtered files.
        read_from_post:   bool  = cfg["buildings_filter"].get(
            "read_from_postprocessed", False
        )

        source_dir = output_dir if read_from_post else predicted_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_files = sorted(source_dir.glob("*.geojson"))
        if not pred_files:
            logger.warning("No .geojson files found in '%s'.", source_dir)
            return 0

        total_removed = 0

        for pred_path in pred_files:
            mask_path = buildings_dir / pred_path.name
            out_path  = output_dir / pred_path.name

            if not mask_path.exists():
                logger.warning(
                    "No buildings mask for '%s' - leaving unchanged.", pred_path.name
                )
                # When reading from postprocessed, file is already in place.
                if not read_from_post:
                    shutil.copy2(pred_path, out_path)
                continue

            buildings_mask = load_mask_wgs84(mask_path)
            kept, removed  = _filter_features(
                pred_path, buildings_mask, max_frac, criterion="area_overlap"
            )
            _write_geojson(pred_path, kept, out_path)

            total_removed += removed
            logger.info(
                "BuildingsFilter '%s': kept %d, removed %d (max_overlap_fraction=%.2f).",
                pred_path.name, len(kept), removed, max_frac,
            )

        return total_removed