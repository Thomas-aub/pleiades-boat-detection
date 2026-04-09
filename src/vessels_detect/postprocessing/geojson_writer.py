"""
src/postprocessing/geojson_writer.py
--------------------------------------
Converts scored :class:`~src.vessels_detect.postprocessing.spatial_filter.OBBBox` detections
into a georeferenced GeoJSON FeatureCollection.

Coordinate pipeline
~~~~~~~~~~~~~~~~~~~~
::

    YOLO prediction  (normalised tile coords)
        ↓  :func:`tile_obb_to_image_pixels`
    Pixel coords in source image
        ↓  :func:`pixels_to_crs`
    Image CRS  (e.g. EPSG:32631 UTM)
        ↓  :func:`crs_to_wgs84`
    WGS-84  →  GeoJSON output

Prediction metadata schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each GeoJSON Feature carries the following ``properties``:

+-----------------+------------------------------------------+------------------+
| Field           | Description                              | Type             |
+=================+==========================================+==================+
| ``label``       | "TP" | "FP" | "FN"                       | str              |
+-----------------+------------------------------------------+------------------+
| ``class_id``    | YOLO class index                         | int              |
+-----------------+------------------------------------------+------------------+
| ``class_name``  | Human-readable class name                | str              |
+-----------------+------------------------------------------+------------------+
| ``confidence``  | Model score (NaN → null for FN)          | float | null     |
+-----------------+------------------------------------------+------------------+
| ``source_tile`` | Tile GeoTIFF stem (absent for FN)        | str | null       |
+-----------------+------------------------------------------+------------------+
| ``source_image``| Source GeoTIFF stem                      | str              |
+-----------------+------------------------------------------+------------------+
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import Polygon, mapping

from src.vessels_detect.postprocessing.spatial_filter import OBBBox

logger = logging.getLogger(__name__)


# =============================================================================
# Coordinate conversion helpers
# =============================================================================

def tile_obb_to_image_pixels(
    corners_norm: np.ndarray,
    x_off: int,
    y_off: int,
    tile_size: int,
) -> np.ndarray:
    """Convert YOLO tile-normalised OBB corners to source-image pixel coords.

    YOLO stores corners relative to the tile; this function maps them back
    to the absolute pixel frame of the original (non-tiled) GeoTIFF.

    Args:
        corners_norm: ``(4, 2)`` array of ``[x_norm, y_norm]`` values in
                      ``[0, 1]`` relative to the tile.
        x_off:        Pixel column offset of the tile's top-left corner.
        y_off:        Pixel row offset of the tile's top-left corner.
        tile_size:    Width and height of the tile in pixels.

    Returns:
        ``(4, 2)`` float array of ``[col, row]`` pixel coordinates in the
        source image frame.
    """
    corners_tile_px = corners_norm * tile_size           # tile-space pixels
    corners_img_px  = corners_tile_px + np.array([[x_off, y_off]])
    return corners_img_px


def pixels_to_crs(
    corners_px: np.ndarray,
    affine: Affine,
) -> np.ndarray:
    """Apply a rasterio Affine transform to convert pixel coords to CRS coords.

    Args:
        corners_px: ``(N, 2)`` array of ``[col, row]`` pixel coordinates.
        affine:     Affine transform of the source GeoTIFF.

    Returns:
        ``(N, 2)`` float array of ``[x_crs, y_crs]`` coordinates.
    """
    corners_crs = np.zeros_like(corners_px, dtype=np.float64)
    for i, (col, row) in enumerate(corners_px):
        x, y = affine * (col, row)
        corners_crs[i] = [x, y]
    return corners_crs


def crs_to_wgs84(
    corners_crs: np.ndarray,
    src_crs: str,
) -> np.ndarray:
    """Reproject corner coordinates from the image CRS to WGS-84.

    Args:
        corners_crs: ``(N, 2)`` array of ``[x, y]`` in *src_crs*.
        src_crs:     EPSG string of the source CRS, e.g. ``"EPSG:32631"``.

    Returns:
        ``(N, 2)`` float array of ``[lon, lat]`` in WGS-84 (EPSG:4326).
        Returns *corners_crs* unchanged when *src_crs* is already WGS-84.
    """
    if src_crs.upper() in ("EPSG:4326", "WGS84"):
        return corners_crs

    import pyproj

    transformer = pyproj.Transformer.from_crs(
        src_crs, "EPSG:4326", always_xy=True
    )
    xs = corners_crs[:, 0]
    ys = corners_crs[:, 1]
    lons, lats = transformer.transform(xs, ys)
    return np.column_stack([lons, lats])


def obb_to_polygon_wgs84(
    corners_norm: np.ndarray,
    x_off: int,
    y_off: int,
    tile_size: int,
    affine: Affine,
    src_crs: str,
) -> Polygon:
    """Full coordinate pipeline: YOLO tile-norm → Shapely WGS-84 polygon.

    Convenience wrapper that chains :func:`tile_obb_to_image_pixels`,
    :func:`pixels_to_crs`, and :func:`crs_to_wgs84`.

    Args:
        corners_norm: ``(4, 2)`` normalised OBB corners from YOLO.
        x_off:        Tile column offset in source image pixels.
        y_off:        Tile row offset in source image pixels.
        tile_size:    Tile dimension in pixels.
        affine:       Source GeoTIFF Affine transform.
        src_crs:      Source CRS string.

    Returns:
        Shapely :class:`~shapely.geometry.Polygon` in WGS-84.
    """
    px  = tile_obb_to_image_pixels(corners_norm, x_off, y_off, tile_size)
    crs = pixels_to_crs(px, affine)
    wgs = crs_to_wgs84(crs, src_crs)
    return Polygon(wgs.tolist())


def gt_to_polygon_wgs84(
    corners_norm: np.ndarray,
    img_width: int,
    img_height: int,
    affine: Affine,
    src_crs: str,
) -> Polygon:
    """Convert global-normalised GT corners to a Shapely WGS-84 polygon.

    Ground-truth labels produced by :mod:`src.preprocessing.steps.annotations`
    are normalised to the full image dimensions (not a tile).

    Args:
        corners_norm: ``(4, 2)`` normalised OBB corners.
        img_width:    Full image width in pixels.
        img_height:   Full image height in pixels.
        affine:       Source GeoTIFF Affine transform.
        src_crs:      Source CRS string.

    Returns:
        Shapely :class:`~shapely.geometry.Polygon` in WGS-84.
    """
    px  = corners_norm * np.array([[img_width, img_height]])
    crs = pixels_to_crs(px, affine)
    wgs = crs_to_wgs84(crs, src_crs)
    return Polygon(wgs.tolist())


# =============================================================================
# Prediction loader  (SAHI / Ultralytics result → OBBBox list)
# =============================================================================

def parse_ultralytics_obb_results(
    results,
    tile_path: Path,
    source_image_stem: str,
    x_off: int,
    y_off: int,
    tile_size: int,
    affine: Affine,
    src_crs: str,
    class_names: Dict[int, str],
) -> List[OBBBox]:
    """Convert an Ultralytics OBB result object into a list of :class:`OBBBox`.

    Iterates over ``results.obb`` boxes, projects each detection from
    tile-normalised coordinates to WGS-84, and wraps the metadata.

    Args:
        results:          Single :class:`ultralytics.engine.results.Results`
                          object from ``model(tile_path)``.
        tile_path:        Path to the tile image (used for ``source_tile``).
        source_image_stem:Stem of the original source GeoTIFF.
        x_off:            Tile column offset in source image pixels.
        y_off:            Tile row offset in source image pixels.
        tile_size:        Tile dimension in pixels.
        affine:           Source GeoTIFF Affine transform.
        src_crs:          Source CRS string.
        class_names:      ``{class_id: class_name}`` mapping.

    Returns:
        List of :class:`OBBBox` in WGS-84, one per detection.
    """
    boxes: List[OBBBox] = []

    if results.obb is None or len(results.obb) == 0:
        return boxes

    # xyxyxyxy: shape (N, 4, 2) in pixel coords relative to the tile image.
    # We need the normalised form: divide by the tile image dimensions.
    h, w = results.orig_shape  # tile image dimensions (may differ from tile_size for edges)

    for i in range(len(results.obb)):
        cls_id = int(results.obb.cls[i].item())
        conf   = float(results.obb.conf[i].item())

        # ``xyxyxyxy`` gives pixel coords; normalise to [0, 1].
        corners_px_tile = results.obb.xyxyxyxy[i].cpu().numpy()  # (4, 2)
        corners_norm    = corners_px_tile / np.array([[w, h]])

        try:
            polygon = obb_to_polygon_wgs84(
                corners_norm, x_off, y_off, tile_size, affine, src_crs
            )
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping malformed OBB at index %d: %s", i, exc)
            continue

        boxes.append(OBBBox(
            polygon=polygon,
            class_id=cls_id,
            confidence=conf,
            source_tile=tile_path.stem,
            source_image=source_image_stem,
            class_name=class_names.get(cls_id, str(cls_id)),
        ))

    return boxes


# =============================================================================
# GeoJSON serialiser
# =============================================================================

def _box_to_feature(box: OBBBox) -> dict:
    """Serialise one :class:`OBBBox` as a GeoJSON Feature dict.

    Args:
        box: Scored :class:`OBBBox` with ``label``, ``class_name``, etc.

    Returns:
        GeoJSON Feature dictionary.
    """
    properties: dict = {
        "label":        box.label,
        "class_id":     box.class_id,
        "class_name":   box.class_name,
        "confidence":   None if math.isnan(box.confidence) else round(box.confidence, 4),
        "source_tile":  box.source_tile,
        "source_image": box.source_image,
    }

    # Remove null-valued optional fields for FN annotations.
    if box.source_tile is None:
        del properties["source_tile"]

    return {
        "type": "Feature",
        "geometry": mapping(box.polygon),
        "properties": properties,
    }


def write_prediction_geojson(
    boxes: Sequence[OBBBox],
    output_path: Path,
    *,
    indent: int = 2,
) -> Path:
    """Serialise all scored boxes to a GeoJSON FeatureCollection.

    The output is a valid RFC 7946 GeoJSON file in WGS-84.  Features are
    ordered: TP first, then FP, then FN - to make inspection easier in QGIS
    or other GIS tools that render features in file order.

    Args:
        boxes:       All :class:`OBBBox` objects (TP + FP + FN combined).
        output_path: Destination ``.geojson`` file path.  Parent directories
                     are created automatically.
        indent:      JSON indentation level (``2`` for readability).

    Returns:
        Resolved path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort: TP → FP → FN.
    order = {"TP": 0, "FP": 1, "FN": 2}
    sorted_boxes = sorted(boxes, key=lambda b: order.get(b.label, 9))

    features = [_box_to_feature(b) for b in sorted_boxes]

    tp_count = sum(1 for b in boxes if b.label == "TP")
    fp_count = sum(1 for b in boxes if b.label == "FP")
    fn_count = sum(1 for b in boxes if b.label == "FN")

    collection = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_features": len(features),
            "TP": tp_count,
            "FP": fp_count,
            "FN": fn_count,
        },
    }

    tmp = output_path.with_suffix(".tmp.geojson")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(collection, fh, indent=indent, ensure_ascii=False)
    tmp.replace(output_path)

    logger.info(
        "GeoJSON written → %s  (TP=%d  FP=%d  FN=%d)",
        output_path, tp_count, fp_count, fn_count,
    )
    return output_path.resolve()
