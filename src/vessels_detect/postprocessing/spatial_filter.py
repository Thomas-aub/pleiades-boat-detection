"""
src/postprocessing/spatial_filter.py
-------------------------------------
Spatial filtering of OBB detections and ground-truth annotations.

Two independent filters are provided:

1.  **Buffer filter** (:func:`keep_inside_buffer`) - retains only the
    detections / GT boxes whose centroid (or a configurable overlap
    fraction) falls inside a polygon mask (``range.geojson``).  This
    constrains evaluation to the operational area of interest and removes
    detections that are geometrically valid but outside the model's scope.

2.  **Building exclusion** (:func:`exclude_building_overlaps`) - removes
    predictions whose overlap with any building polygon exceeds a
    configurable IoU / intersection-over-prediction threshold.  Ground truth
    is never mutated by this filter; only predicted boxes are pruned.

Both functions operate on :class:`OBBBox` instances - thin dataclasses
that carry the minimal geometry needed by all downstream consumers (scorer,
GeoJSON writer).

Coordinate system note
~~~~~~~~~~~~~~~~~~~~~~
:class:`OBBBox` polygons are always expressed in **WGS-84 (EPSG:4326)**.
Mask GeoJSON files (``range.geojson``, ``buildings.geojson``) are also in
WGS-84 per RFC 7946, so no reprojection is required before calling these
filters.

If you ever need to load a mask into a projected CRS (e.g. for pixel-space
operations before coordinate conversion), use
:func:`~src.vessels_detect.utils.crs.load_mask_in_crs` directly.

Mask loading
~~~~~~~~~~~~
Mask loading is **delegated entirely** to
:mod:`src.vessels_detect.utils.crs`, which is the single authoritative
place for CRS-aware GeoJSON loading in this project.  The re-exports
below are kept for backwards compatibility so that existing imports of
``load_mask_union`` from this module continue to work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from shapely.geometry import MultiPolygon, Polygon

# ---------------------------------------------------------------------------
# CRS utilities - single source of truth for mask loading & reprojection
# ---------------------------------------------------------------------------
from src.vessels_detect.utils.crs import (
    load_mask_wgs84,
    load_mask_in_crs,
    reproject_geometry as _reproject_geometry,
)

logger = logging.getLogger(__name__)

# Intersection-over-prediction threshold for building exclusion.
# A box is dropped when (box ∩ building) / box_area ≥ this value.
_DEFAULT_IOP_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
# Old call sites that imported ``load_mask_union`` from this module will
# still work.  New code should import from src.vessels_detect.utils.crs
# directly.
def load_mask_union(
    geojson_path: Path,
    target_crs: Optional[str] = None,
) -> MultiPolygon:
    """Load a GeoJSON mask and return the union of all its polygons.

    .. deprecated::
        Import :func:`~src.vessels_detect.utils.crs.load_mask_wgs84` or
        :func:`~src.vessels_detect.utils.crs.load_mask_in_crs` directly.
        This shim will be removed in a future release.

    Args:
        geojson_path: Path to a ``.geojson`` or ``.json`` file in
                      EPSG:4326 (RFC 7946).
        target_crs:   When ``None`` or ``"EPSG:4326"`` the mask is returned
                      in WGS-84.  Any other value reprojects the union to
                      that CRS.

    Returns:
        :class:`~shapely.geometry.MultiPolygon` union.
    """
    import warnings
    warnings.warn(
        "load_mask_union() is deprecated. "
        "Use src.vessels_detect.utils.crs.load_mask_wgs84() or "
        "load_mask_in_crs() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if target_crs is None:
        return load_mask_wgs84(geojson_path)
    return load_mask_in_crs(geojson_path, target_crs=target_crs)


# =============================================================================
# Core data type
# =============================================================================

@dataclass
class OBBBox:
    """One oriented bounding box with its associated metadata.

    This is the common currency shared between the predictor, the filters,
    the scorer, and the GeoJSON writer.  Geometry is stored as a Shapely
    :class:`~shapely.geometry.Polygon` in **WGS-84 (EPSG:4326)** so that
    spatial operations across all pipeline stages use a single CRS.

    Attributes:
        polygon:      Shapely polygon of the OBB in WGS-84.
        class_id:     YOLO class index.
        confidence:   Model confidence in ``[0, 1]``.  ``float("nan")``
                      for ground-truth annotations.
        source_tile:  Stem of the tile GeoTIFF that produced this prediction.
                      ``None`` for ground-truth annotations.
        source_image: Stem of the original GeoTIFF (before tiling).
        label:        ``"TP"`` | ``"FP"`` | ``"FN"`` - populated by the scorer.
        class_name:   Human-readable class name - populated after scoring.
    """

    polygon:      Polygon
    class_id:     int
    confidence:   float         = float("nan")
    source_tile:  Optional[str] = None
    source_image: str           = ""
    label:        str           = ""     # assigned by scorer
    class_name:   str           = ""     # assigned by scorer

    @property
    def centroid(self) -> Tuple[float, float]:
        """Centroid of the OBB as ``(x, y)`` in WGS-84."""
        c = self.polygon.centroid
        return (c.x, c.y)


# =============================================================================
# Filter 1 - buffer / operational-area mask
# =============================================================================

def keep_inside_buffer(
    boxes: Sequence[OBBBox],
    buffer_mask: MultiPolygon,
    *,
    min_intersection_frac: float = 0.50,
) -> List[OBBBox]:
    """Retain only boxes whose overlap with *buffer_mask* meets the threshold.

    A box passes the filter when::

        (box ∩ buffer_mask).area / box.area  ≥  min_intersection_frac

    Setting ``min_intersection_frac=0.0`` keeps any box that merely touches
    the buffer.  Setting it to ``1.0`` requires the box to be fully contained.

    This filter is applied identically to both predictions and ground-truth
    annotations so that the evaluation is restricted to the operational area.

    Both *boxes* and *buffer_mask* must be in the **same CRS** (WGS-84 by
    default throughout this pipeline).

    Args:
        boxes:                  Sequence of :class:`OBBBox` to filter.
        buffer_mask:            Union polygon of the operational area.
        min_intersection_frac:  Fraction of the box area that must lie inside
                                the buffer for the box to be retained.

    Returns:
        Filtered list; original order is preserved.
    """
    kept:    List[OBBBox] = []
    dropped: int          = 0

    for box in boxes:
        poly = box.polygon
        if poly.is_empty or poly.area < 1e-12:
            dropped += 1
            continue

        intersection = poly.intersection(buffer_mask)
        frac = intersection.area / poly.area

        if frac >= min_intersection_frac:
            kept.append(box)
        else:
            dropped += 1

    logger.debug(
        "Buffer filter: %d / %d boxes retained (threshold=%.2f).",
        len(kept), len(boxes), min_intersection_frac,
    )
    return kept


# =============================================================================
# Filter 2 - building exclusion
# =============================================================================

def exclude_building_overlaps(
    predictions: Sequence[OBBBox],
    buildings_mask: MultiPolygon,
    *,
    iop_threshold: float = _DEFAULT_IOP_THRESHOLD,
) -> List[OBBBox]:
    """Remove predicted boxes whose overlap with *buildings_mask* is too large.

    The **intersection-over-prediction** (IoP) metric is used rather than
    standard IoU because buildings are typically much larger than vessels;
    using IoU would almost never trigger the filter.  IoP measures what
    fraction of the *prediction* box is occupied by a building::

        IoP = (prediction ∩ buildings_mask).area / prediction.area

    Predictions with ``IoP ≥ iop_threshold`` are removed.  Ground-truth
    annotations are **never** passed through this filter - FN boxes on
    buildings are retained so the recall computation is not artificially
    inflated.

    Both *predictions* and *buildings_mask* must be in the **same CRS**
    (WGS-84 by default throughout this pipeline).

    Args:
        predictions:    Sequence of predicted :class:`OBBBox` to filter.
        buildings_mask: Union polygon of all building footprints.
        iop_threshold:  IoP above which a prediction is dropped
                        (default ``0.30``).

    Returns:
        Filtered predictions; original order is preserved.
    """
    kept:    List[OBBBox] = []
    dropped: int          = 0

    for pred in predictions:
        poly = pred.polygon
        if poly.is_empty or poly.area < 1e-12:
            dropped += 1
            continue

        intersection = poly.intersection(buildings_mask)
        iop = intersection.area / poly.area

        if iop >= iop_threshold:
            dropped += 1
            logger.debug(
                "Dropped prediction (IoP=%.3f ≥ %.3f) at centroid %s.",
                iop, iop_threshold, pred.centroid,
            )
        else:
            kept.append(pred)

    logger.info(
        "Building exclusion: %d prediction(s) dropped (IoP≥%.2f), %d kept.",
        dropped, iop_threshold, len(kept),
    )
    return kept