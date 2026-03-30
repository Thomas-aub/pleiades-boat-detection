"""
src/vessels_detect/inference/postprocess.py
-----------------------------
Global Non-Maximum Suppression (NMS) for tiled satellite-imagery inference.

**The tile-overlap problem**
    Adjacent tiles share an ``overlap`` pixel border (configured in
    ``configs/data_prep.yaml``).  A boat sitting on a tile boundary will
    therefore be detected by two or more tiles.  YOLO's internal NMS
    operates only within a single tile, so these duplicates survive into the
    raw prediction stream.

**Solution: project to a shared coordinate space, then suppress globally**
    1. Every raw prediction from :mod:`src.inference.predictor` carries the
       tile's Affine transform (pixel → CRS).
    2. :func:`project_to_crs` converts the 4 pixel-space OBB corners of
       each detection into the tile's native CRS (e.g., UTM).
    3. :func:`global_nms` runs a *greedy area-overlap* NMS pass on all
       detections projected into that shared CRS.  Detections are first
       sorted by descending confidence; then any subsequent detection whose
       rotated-polygon IoU with an already-accepted detection exceeds
       ``iou_threshold`` is suppressed.

**Coordinate invariance**
    Because all tiles from the same source GeoTIFF share the same CRS, the
    projected corners live in a single consistent metric coordinate system
    (metres or degrees), making inter-tile overlap comparison exact.

Typical usage::

    from src.inference.postprocess import GlobalNMS

    nms = GlobalNMS(iou_threshold=0.30)
    kept = nms.run(tile_predictions)   # list[GlobalDetection]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Polygon

from src.vessels_detect.inference.predictor import TilePrediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GlobalDetection:
    """A single detection after global NMS, carrying full spatial context.

    Attributes:
        class_id: Integer YOLO class index.
        confidence: Model confidence score in [0, 1].
        crs_corners: OBB corners in the tile's native CRS (e.g. UTM).
            Shape ``(4, 2)`` — ``(easting/x, northing/y)`` pairs.
        tile_path: Path to the source GeoTIFF tile.
        crs: CRS string (e.g. ``"EPSG:32632"``).
    """

    class_id:    int
    confidence:  float
    crs_corners: np.ndarray   # (4, 2)  in tile's native CRS
    tile_path:   Path
    crs:         str


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rotated_iou_crs(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Compute the rotated-polygon IoU of two OBBs in CRS space.

    Uses Shapely to safely handle massive float64 UTM coordinates 
    without the truncation errors caused by OpenCV's float32 casting.

    Args:
        pts_a: Shape ``(4, 2)`` — corners of OBB A in CRS coordinates.
        pts_b: Shape ``(4, 2)`` — corners of OBB B in CRS coordinates.

    Returns:
        IoU value in ``[0, 1]``.
    """
    poly_a = Polygon(pts_a)
    poly_b = Polygon(pts_b)

    # Fast bounds check before heavy intersection math
    if not poly_a.intersects(poly_b):
        return 0.0

    inter_area = poly_a.intersection(poly_b).area
    union_area = poly_a.area + poly_b.area - inter_area
    
    return float(inter_area / (union_area + 1e-12))


def project_corners_to_crs(
    pixel_corners: np.ndarray,
    tile_transform: Affine,
) -> np.ndarray:
    """Convert pixel-space OBB corners to the tile's native CRS.

    Applies the tile Affine transform ``T`` such that::

        (easting, northing) = T * (col, row)

    Args:
        pixel_corners: Shape ``(4, 2)`` — ``(col, row)`` pairs.
        tile_transform: Affine transform from the source GeoTIFF.

    Returns:
        Array of shape ``(4, 2)`` — ``(x, y)`` pairs in the tile's CRS.
    """
    crs_corners = np.array(
        [tile_transform * (float(c[0]), float(c[1])) for c in pixel_corners],
        dtype=np.float64,
    )
    return crs_corners


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GlobalNMS:
    """Applies global NMS across all tiles to remove border duplicates.

    Detections are sorted by descending confidence.  A detection is
    *suppressed* if its rotated-polygon IoU with any already-accepted
    detection (of the **same class**) exceeds ``iou_threshold``.

    Cross-class suppression is intentionally disabled: a pirogue and a
    motorboat overlapping in the same pixel region are genuinely different
    objects and should both be kept.

    Args:
        iou_threshold: IoU above which a lower-confidence duplicate is
            suppressed.  Typical range: 0.20 – 0.50.
    """

    def __init__(self, iou_threshold: float = 0.30) -> None:
        if not 0.0 < iou_threshold < 1.0:
            raise ValueError(
                f"iou_threshold must be in (0, 1), got {iou_threshold}."
            )
        self._iou_thr = iou_threshold

    def run(self, predictions: Sequence[TilePrediction]) -> List[GlobalDetection]:
        """Apply global NMS to all tile predictions and return kept detections.

        Args:
            predictions: Iterable of :class:`~src.inference.predictor.TilePrediction`
                objects as yielded by :class:`~src.inference.predictor.YoloPredictor`.

        Returns:
            List of :class:`GlobalDetection` objects after suppression, sorted
            by descending confidence.
        """
        # ── 1. Unpack all detections into a flat list ──────────────────
        raw: List[GlobalDetection] = []
        for tile_pred in predictions:
            if not tile_pred.has_detections:
                continue

            for i in range(tile_pred.n_detections):
                crs_pts = project_corners_to_crs(
                    tile_pred.pixel_corners[i],   # (4, 2)
                    tile_pred.tile_transform,
                )
                raw.append(GlobalDetection(
                    class_id=int(tile_pred.class_ids[i]),
                    confidence=float(tile_pred.confidences[i]),
                    crs_corners=crs_pts,
                    tile_path=tile_pred.tile_path,
                    crs=tile_pred.crs,
                ))

        if not raw:
            logger.info("GlobalNMS: no detections to process.")
            return []

        logger.info(
            "GlobalNMS: %d raw detection(s) across %d tile(s).  "
            "Running per-class suppression (IoU threshold=%.2f) …",
            len(raw), len({d.tile_path for d in raw}), self._iou_thr,
        )

        # ── 2. Sort by descending confidence ──────────────────────────
        raw.sort(key=lambda d: d.confidence, reverse=True)

        # ── 3. Greedy suppression (per class) ─────────────────────────
        kept: List[GlobalDetection] = []
        suppressed_flags = [False] * len(raw)

        for i, det_i in enumerate(raw):
            if suppressed_flags[i]:
                continue

            kept.append(det_i)

            for j in range(i + 1, len(raw)):
                if suppressed_flags[j]:
                    continue
                
                det_j = raw[j]

                # CRITICAL BUG FIX: Ensure we only suppress duplicates of the SAME class
                if det_i.class_id != det_j.class_id:
                    continue

                iou = rotated_iou_crs(det_i.crs_corners, det_j.crs_corners)
                if iou >= self._iou_thr:
                    suppressed_flags[j] = True

        n_suppressed = len(raw) - len(kept)
        logger.info(
            "GlobalNMS complete.  Kept: %d  |  Suppressed: %d",
            len(kept), n_suppressed,
        )
        return kept