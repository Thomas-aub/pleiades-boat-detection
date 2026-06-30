"""
src/vessels_detect/preprocessing/label_conversion.py
------------------------------------------------------
Step 2 - Label conversion: GeoJSON OBB -> YOLO OBB.

Converts the GeoJSON oriented-bounding-box annotations matching a raw image
into a YOLO OBB ``.txt`` file, normalised against the *enhanced* image's
pixel dimensions (not a tile) - the correct frame for SAHI-based inference.

Coordinate flow per annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    GeoJSON exterior ring (WGS 84)
        -> minimum_rotated_rectangle (Shapely)
        -> reproject to image CRS (pyproj)
        -> apply inverse image Affine -> pixel (col, row)
        -> enforce minimum side length
        -> normalise by (image_width, image_height) -> [0, 1]
        -> clamp to image boundary
        -> "class_id x1 y1 x2 y2 x3 y3 x4 y4"

This step only reads the enhanced image's small metadata header (CRS,
Affine, width/height) - never the pixel array - so it has a negligible
memory footprint regardless of image size.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyproj
import rasterio
from affine import Affine
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------

def _load_features(geojson_path: Path) -> List[dict]:
    with open(geojson_path) as fh:
        return json.load(fh).get("features", [])


def _obb_corners_wgs84(feature: dict) -> Optional[List[Tuple[float, float]]]:
    """Extract 4 corners (lon, lat) from a GeoJSON polygon, fixing it via
    Shapely's minimum_rotated_rectangle if it isn't already a clean quad."""
    try:
        coords = feature["geometry"]["coordinates"][0]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                return None

        pts = list(poly.minimum_rotated_rectangle.exterior.coords)
        if pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) != 4:
            return None
        return [(float(p[0]), float(p[1])) for p in pts]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not extract OBB: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Coordinate projection
# ---------------------------------------------------------------------------

def _project_corners(
    corners_wgs84: List[Tuple[float, float]],
    transformer: Optional[pyproj.Transformer],
) -> List[Tuple[float, float]]:
    if transformer is None:
        return corners_wgs84
    xs, ys = [c[0] for c in corners_wgs84], [c[1] for c in corners_wgs84]
    tx, ty = transformer.transform(xs, ys)
    return list(zip(tx, ty))


def _corners_to_pixel(
    corners_crs: List[Tuple[float, float]], inv_transform: Affine
) -> List[Tuple[float, float]]:
    return [inv_transform * (x, y) for x, y in corners_crs]


def _enforce_min_side(
    corners_px: List[Tuple[float, float]], min_size: float
) -> List[Tuple[float, float]]:
    """Symmetrically elongate degenerate OBB axes to a minimum pixel size."""
    p0, p1, p2, p3 = corners_px
    ax, bx = (p1[0] - p0[0], p1[1] - p0[1]), (p3[0] - p0[0], p3[1] - p0[1])
    len_a, len_b = math.hypot(*ax), math.hypot(*bx)
    ua = (ax[0] / len_a, ax[1] / len_a) if len_a > 1e-9 else (1.0, 0.0)
    ub = (bx[0] / len_b, bx[1] / len_b) if len_b > 1e-9 else (0.0, 1.0)
    cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0
    cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0
    half_a, half_b = max(len_a / 2.0, min_size / 2.0), max(len_b / 2.0, min_size / 2.0)
    return [
        (cx - ua[0]*half_a - ub[0]*half_b, cy - ua[1]*half_a - ub[1]*half_b),
        (cx + ua[0]*half_a - ub[0]*half_b, cy + ua[1]*half_a - ub[1]*half_b),
        (cx + ua[0]*half_a + ub[0]*half_b, cy + ua[1]*half_a + ub[1]*half_b),
        (cx - ua[0]*half_a + ub[0]*half_b, cy - ua[1]*half_a + ub[1]*half_b),
    ]


def _normalise_global(
    corners_px: List[Tuple[float, float]], img_width: int, img_height: int
) -> List[Tuple[float, float]]:
    return [
        (max(0.0, min(1.0, col / img_width)), max(0.0, min(1.0, row / img_height)))
        for col, row in corners_px
    ]


def _to_yolo_line(class_id: int, norm: List[Tuple[float, float]]) -> str:
    coords = " ".join(f"{v:.6f}" for pt in norm for v in pt)
    return f"{class_id} {coords}"


def _find_geojson(raw_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".geojson", ".json"):
        candidate = raw_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Public entry point - used by scripts/preprocessing.py
# ---------------------------------------------------------------------------

def convert_labels(
    enhanced_image_path: Path,
    raw_dir: Path,
    out_dir: Path,
    cfg: Dict,
) -> Path:
    """Convert the GeoJSON OBBs matching one enhanced image into a YOLO ``.txt``.

    Args:
        enhanced_image_path: Path to the enhanced GeoTIFF (output of
            ``image_enhancement.enhance_image``). Only its header (CRS,
            Affine, width/height) is read - never the pixel array.
        raw_dir: Directory containing the source ``.geojson``/``.json``
            files, matched to the image by filename stem.
        out_dir: Directory where the ``.txt`` label file is written.
        cfg: Full resolved pipeline config (uses ``cfg["annotations"]``).

    Returns:
        Path to the written YOLO OBB label file (empty file = background
        image, as required by the YOLO dataloader).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    params = cfg["annotations"]
    min_visible = params.get("min_visible", 0.10)
    min_size_px = params.get("min_size_px", 2.0)
    class_map = {int(k): int(v) for k, v in params.get("class_map", {}).items()}
    skip_classes = set(int(x) for x in params.get("skip_classes", []))

    label_path = out_dir / f"{enhanced_image_path.stem}.txt"
    geojson_path = _find_geojson(raw_dir, enhanced_image_path.stem)

    if geojson_path is None:
        logger.warning("  [labels] No GeoJSON for '%s' - writing empty label.", enhanced_image_path.name)
        label_path.write_text("")
        return label_path

    features = _load_features(geojson_path)

    with rasterio.open(enhanced_image_path) as ds:
        crs_str, img_transform = str(ds.crs), ds.transform
        img_width, img_height = ds.width, ds.height

    inv_transform = ~img_transform
    transformer = None if crs_str == "EPSG:4326" else pyproj.Transformer.from_crs(
        "EPSG:4326", crs_str, always_xy=True
    )

    tl = img_transform * (0, 0)
    tr = img_transform * (img_width, 0)
    br = img_transform * (img_width, img_height)
    bl = img_transform * (0, img_height)
    img_poly = Polygon([tl, tr, br, bl])

    lines: List[str] = []
    # GDAL/QGIS often serialise a missing int32 field as this sentinel
    # instead of JSON null - treat it the same as "no class_id".
    _INT32_NODATA = -2147483647

    for feature in features:
        raw_cls_value = feature.get("properties", {}).get("class_id")
        if raw_cls_value is None or int(raw_cls_value) == _INT32_NODATA:
            logger.debug("  [labels] Feature has no class_id - skipping.")
            continue

        raw_cls = int(raw_cls_value)
        if raw_cls in skip_classes:
            continue

        yolo_cls = class_map.get(raw_cls)
        if yolo_cls is None:
            logger.warning("  [labels] Unknown class_id=%d - skipping.", raw_cls)
            continue

        corners_wgs84 = _obb_corners_wgs84(feature)
        if corners_wgs84 is None:
            continue

        corners_crs = _project_corners(corners_wgs84, transformer)
        ann_poly = Polygon(corners_crs)
        if not img_poly.intersects(ann_poly):
            continue

        vis_frac = (
            img_poly.intersection(ann_poly).area / ann_poly.area
            if ann_poly.area > 1e-12 else 0.0
        )
        if vis_frac < min_visible:
            continue

        corners_px = _corners_to_pixel(corners_crs, inv_transform)
        corners_px = _enforce_min_side(corners_px, min_size_px)
        norm = _normalise_global(corners_px, img_width, img_height)

        if Polygon(norm).area < 1e-12:
            continue

        lines.append(_to_yolo_line(yolo_cls, norm))

    with open(label_path, "w") as fh:
        if lines:
            fh.write("\n".join(lines) + "\n")
        # Empty file = background image.

    logger.debug("  [labels] %s -> %d annotation(s)", enhanced_image_path.name, len(lines))
    return label_path