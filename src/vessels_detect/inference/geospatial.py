"""
src/inference/geospatial.py
----------------------------
Converts post-NMS detections into a standard GeoJSON FeatureCollection
with WGS 84 (EPSG:4326) coordinates.

**Coordinate pipeline**
    Each :class:`~src.inference.postprocess.GlobalDetection` already carries
    its OBB corners projected into the tile's native CRS (e.g. UTM Zone 32N).
    This module applies a second projection step:

        CRS corners  →  pyproj  →  (longitude, latitude) in EPSG:4326

    Because all tiles from the same source image share the same CRS, one
    :class:`pyproj.Transformer` is built per unique CRS and cached for
    reuse.

**Output format**
    The output is a valid ``GeoJSON FeatureCollection`` where every feature:

    - has a ``Polygon`` geometry (the 4-corner OBB, closed);
    - carries ``class_id``, ``class_name``, ``confidence``, and
      ``source_tile`` in its ``properties`` dict.

    The file can be loaded directly in QGIS, ArcGIS, or any GIS tool.

Typical usage::

    from src.inference.geospatial import GeoJSONExporter

    exporter = GeoJSONExporter(class_names={0: "Pirogue", 1: "Motorboat"})
    exporter.export(detections, output_path=Path("predictions.geojson"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyproj

from src.inference.postprocess import GlobalDetection

logger = logging.getLogger(__name__)

# EPSG:4326 is the standard output CRS for GeoJSON (RFC 7946).
_TARGET_CRS = "EPSG:4326"


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

class _TransformerCache:
    """Caches :class:`pyproj.Transformer` instances by source CRS string.

    Building a Transformer is not free (involves CRS parsing).  Caching
    avoids redundant construction when thousands of detections share the
    same source CRS.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, pyproj.Transformer] = {}

    def get(self, src_crs: str) -> pyproj.Transformer:
        """Return a cached or newly built Transformer for *src_crs* → WGS 84.

        Args:
            src_crs: Source CRS identifier string (as returned by
                ``rasterio.DatasetReader.crs``).

        Returns:
            A :class:`pyproj.Transformer` with ``always_xy=True`` so that
            coordinates are always (easting/longitude, northing/latitude).

        Raises:
            pyproj.exceptions.CRSError: If *src_crs* is not parseable.
        """
        if src_crs not in self._cache:
            self._cache[src_crs] = pyproj.Transformer.from_crs(
                src_crs, _TARGET_CRS, always_xy=True
            )
        return self._cache[src_crs]


_TRANSFORMER_CACHE = _TransformerCache()


def project_to_wgs84(
    crs_corners: np.ndarray,
    src_crs: str,
) -> List[List[float]]:
    """Project OBB corners from *src_crs* to WGS 84 (lon, lat).

    Args:
        crs_corners: Shape ``(4, 2)`` — ``(x, y)`` corner pairs in
            the tile's native CRS.
        src_crs: Source CRS string.

    Returns:
        List of 5 ``[longitude, latitude]`` pairs (the polygon is closed
        by repeating the first point, as required by GeoJSON RFC 7946).

    Raises:
        pyproj.exceptions.CRSError: If *src_crs* cannot be parsed.
    """
    transformer = _TRANSFORMER_CACHE.get(src_crs)
    wgs84_coords: List[List[float]] = []

    for pt in crs_corners:
        lon, lat = transformer.transform(float(pt[0]), float(pt[1]))
        wgs84_coords.append([round(lon, 8), round(lat, 8)])

    # Close the GeoJSON ring.
    wgs84_coords.append(wgs84_coords[0])
    return wgs84_coords


# ---------------------------------------------------------------------------
# GeoJSON serialisation
# ---------------------------------------------------------------------------

def _build_feature(
    detection: GlobalDetection,
    class_names: Dict[int, str],
) -> dict:
    """Serialise one :class:`~src.inference.postprocess.GlobalDetection` as a GeoJSON Feature.

    Args:
        detection: A post-NMS detection with CRS-projected corners.
        class_names: Mapping ``{class_id: class_name}``.

    Returns:
        A GeoJSON ``Feature`` dictionary with a ``Polygon`` geometry and
        relevant ``properties``.
    """
    try:
        coords_wgs84 = project_to_wgs84(detection.crs_corners, detection.crs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to project detection from '%s' to WGS84: %s — skipping.",
            detection.tile_path.name, exc,
        )
        return {}

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords_wgs84],
        },
        "properties": {
            "class_id":    detection.class_id,
            "class_name":  class_names.get(detection.class_id, str(detection.class_id)),
            "confidence":  round(float(detection.confidence), 4),
            "source_tile": detection.tile_path.name,
            "crs":         detection.crs,
        },
    }


def _build_feature_collection(features: List[dict]) -> dict:
    """Wrap a list of GeoJSON Feature dicts in a FeatureCollection.

    Args:
        features: List of GeoJSON Feature dictionaries.

    Returns:
        A valid GeoJSON FeatureCollection dictionary.
    """
    return {
        "type": "FeatureCollection",
        "features": [f for f in features if f],  # drop empty dicts from errors
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GeoJSONExporter:
    """Converts post-NMS detections to a GeoJSON FeatureCollection file.

    Args:
        class_names: Mapping ``{class_id: class_name}`` used to populate
            the ``class_name`` property of each GeoJSON feature.
            Typically obtained from ``YoloPredictor.class_names``.
    """

    def __init__(self, class_names: Dict[int, str]) -> None:
        self._class_names = class_names

    def export(
        self,
        detections: List[GlobalDetection],
        output_path: Path,
        indent: Optional[int] = 2,
    ) -> Path:
        """Project detections to WGS 84 and write a GeoJSON file.

        Args:
            detections: List of :class:`~src.inference.postprocess.GlobalDetection`
                objects returned by :meth:`~src.inference.postprocess.GlobalNMS.run`.
            output_path: Destination path for the ``.geojson`` file.
                Parent directories are created automatically.
            indent: JSON indentation level.  ``None`` produces compact
                single-line JSON (smaller file, harder to diff).

        Returns:
            The resolved absolute path to the written file.

        Raises:
            OSError: If the file cannot be written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Exporting %d detection(s) to '%s' …", len(detections), output_path
        )

        features = [
            _build_feature(det, self._class_names)
            for det in detections
        ]
        collection = _build_feature_collection(features)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(collection, fh, indent=indent, ensure_ascii=False)

        n_written = len(collection["features"])
        n_skipped = len(detections) - n_written
        logger.info(
            "GeoJSON written: %d feature(s)%s",
            n_written,
            f"  ({n_skipped} skipped due to projection errors)" if n_skipped else "",
        )
        logger.info("Output: %s", output_path.resolve())
        return output_path.resolve()

    def to_dict(self, detections: List[GlobalDetection]) -> dict:
        """Return the GeoJSON FeatureCollection as a Python dictionary.

        Useful for in-memory inspection or passing to mapping libraries
        (e.g., ``folium``, ``geopandas``).

        Args:
            detections: List of post-NMS detections.

        Returns:
            A GeoJSON FeatureCollection dictionary.
        """
        features = [
            _build_feature(det, self._class_names)
            for det in detections
        ]
        return _build_feature_collection(features)
