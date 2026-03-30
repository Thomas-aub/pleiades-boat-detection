"""
src/vessels_detect/preprocessing/steps/annotations.py
------------------------------------------------------
Stage 3 — Global Annotation Conversion.

Converts GeoJSON oriented bounding-box (OBB) annotations into YOLO OBB
``.txt`` label files, with coordinates normalised to the **global** processed
image dimensions — not to a tile.  This is the correct reference frame for
SAHI-based inference, where YOLO sees the whole image and SAHI post-processes
overlapping slices.

Coordinate flow per annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    GeoJSON exterior ring  (WGS 84 / EPSG:4326)
        ↓  minimum_rotated_rectangle  (Shapely)
        ↓  reproject to image CRS  (pyproj Transformer)
        ↓  apply inverse image Affine  → pixel (col, row)
        ↓  enforce minimum side length
        ↓  normalise by (image_width, image_height)  → [0, 1]
        ↓  clamp corners to image boundary
        ↓  write YOLO OBB line:  class_id x1 y1 x2 y2 x3 y3 x4 y4

Key difference from the old per-tile converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Normalisation denominator is ``(image_width, image_height)`` — the full
  image, not a 320 px tile.
* ``min_visible`` checks visibility against the full image footprint, not a
  tile footprint (practically only clips objects on the sensor swath edge).
* No tile grouping logic; one GeoJSON ↔ one processed GeoTIFF (matched by
  stem).

Output
~~~~~~
One ``.txt`` per processed image in ``labels_dir``.  Background images
(no overlapping annotations) receive an empty file, as required by YOLO.

Typical usage (via manager)::

    from src.vessels_detect.preprocessing.steps.annotations import AnnotationStep
    step = AnnotationStep()
    step.run(cfg)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pyproj
import rasterio
from affine import Affine
from shapely.geometry import Polygon

from src.vessels_detect.preprocessing.steps.base import BaseStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnnotationConfig:
    """Hyperparameters for :class:`AnnotationStep`.

    Attributes:
        min_visible: Minimum fraction of an OBB's area that must lie
            within the image boundary for it to be included.  Primarily
            catches annotations on the sensor swath edge.
        min_size_px: Minimum OBB side length in pixels (output image space).
            Degenerate boxes are symmetrically elongated to this size.
        class_map: GeoJSON ``class_id`` → YOLO class index remapping.
        skip_classes: Set of GeoJSON ``class_id`` values to discard.
    """

    min_visible:   float          = 0.10
    min_size_px:   float          = 2.0
    class_map:     Dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4}
    )
    skip_classes:  Set[int]       = field(default_factory=lambda: {9})

    @classmethod
    def from_dict(cls, cfg: dict) -> "AnnotationConfig":
        """Construct from a YAML section dictionary.

        Args:
            cfg: Dictionary with keys matching the dataclass field names.
                ``class_map`` keys and values are coerced to ``int``.

        Returns:
            A populated :class:`AnnotationConfig` instance.
        """
        raw_cm = cfg.get("class_map", {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4})
        return cls(
            min_visible=cfg.get("min_visible", 0.10),
            min_size_px=cfg.get("min_size_px", 2.0),
            class_map={int(k): int(v) for k, v in raw_cm.items()},
            skip_classes=set(int(x) for x in cfg.get("skip_classes", [9])),
        )


# ---------------------------------------------------------------------------
# GeoJSON helpers
# ---------------------------------------------------------------------------

def _load_features(geojson_path: Path) -> List[dict]:
    """Parse a GeoJSON file and return its feature list.

    Args:
        geojson_path: Path to a ``.geojson`` or ``.json`` annotation file.

    Returns:
        List of GeoJSON feature dictionaries.  Empty list if no
        ``"features"`` key is present.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the content is not valid JSON.
    """
    with open(geojson_path) as fh:
        gj = json.load(fh)
    return gj.get("features", [])


def _obb_corners_wgs84(feature: dict) -> Optional[List[Tuple[float, float]]]:
    """Extract exactly 4 corner coordinates from a GeoJSON polygon feature.

    If the polygon has more or fewer than 4 vertices, Shapely's
    ``minimum_rotated_rectangle`` is used to compute a valid OBB.  Invalid
    rings are repaired with ``buffer(0)``.

    Args:
        feature: GeoJSON Feature dictionary with a ``Polygon`` geometry.

    Returns:
        List of 4 ``(lon, lat)`` tuples in WGS 84 (EPSG:4326), or ``None``
        if the geometry is absent, malformed, or cannot be fixed.
    """
    try:
        coords = feature["geometry"]["coordinates"][0]
        poly   = Polygon(coords)

        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                return None

        obb  = poly.minimum_rotated_rectangle
        pts  = list(obb.exterior.coords)
        if pts[0] == pts[-1]:
            pts = pts[:-1]

        if len(pts) != 4:
            logger.debug("OBB has %d vertices (expected 4) — skipping.", len(pts))
            return None

        return [(float(p[0]), float(p[1])) for p in pts]

    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not extract OBB from feature: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Coordinate projection helpers
# ---------------------------------------------------------------------------

def _build_transformer(
    src_crs: str,
    dst_crs: str,
) -> Optional[pyproj.Transformer]:
    """Build a pyproj Transformer from *src_crs* to *dst_crs*.

    Args:
        src_crs: Source CRS string (e.g. ``"EPSG:4326"``).
        dst_crs: Destination CRS string (e.g. ``"EPSG:32631"``).

    Returns:
        A :class:`pyproj.Transformer` instance, or ``None`` if both CRS
        strings are identical (no reprojection needed).
    """
    if src_crs == dst_crs:
        return None
    return pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    )


def _project_corners(
    corners_wgs84: List[Tuple[float, float]],
    transformer: Optional[pyproj.Transformer],
) -> List[Tuple[float, float]]:
    """Reproject OBB corners from WGS 84 to the image CRS.

    Args:
        corners_wgs84: List of 4 ``(lon, lat)`` tuples.
        transformer: Pyproj Transformer.  ``None`` returns *corners_wgs84*
            unchanged.

    Returns:
        List of 4 ``(x, y)`` tuples in the image's native CRS.
    """
    if transformer is None:
        return corners_wgs84
    xs = [c[0] for c in corners_wgs84]
    ys = [c[1] for c in corners_wgs84]
    tx, ty = transformer.transform(xs, ys)
    return list(zip(tx, ty))


# ---------------------------------------------------------------------------
# Pixel-space helpers
# ---------------------------------------------------------------------------

def _corners_to_pixel(
    corners_crs: List[Tuple[float, float]],
    inv_transform: Affine,
) -> List[Tuple[float, float]]:
    """Apply the inverse Affine transform to project corners into pixel space.

    Args:
        corners_crs: List of 4 ``(x, y)`` coordinate pairs in the image CRS.
        inv_transform: Inverse of the image's Affine transform.

    Returns:
        List of 4 ``(col, row)`` pixel-coordinate tuples.
    """
    return [inv_transform * (x, y) for x, y in corners_crs]


def _enforce_min_side(
    corners_px: List[Tuple[float, float]],
    min_size: float,
) -> List[Tuple[float, float]]:
    """Symmetrically elongate OBB axes shorter than *min_size*.

    The OBB centre is kept fixed; both axes are elongated independently.
    This prevents degenerate zero-area boxes that YOLO would reject.

    Args:
        corners_px: 4 ``(col, row)`` pixel-coordinate tuples (consecutive
            vertices share an edge).
        min_size: Minimum allowable side length in pixels.

    Returns:
        4 ``(col, row)`` tuples, potentially adjusted.
    """
    p0, p1, p2, p3 = corners_px

    ax = (p1[0] - p0[0], p1[1] - p0[1])
    bx = (p3[0] - p0[0], p3[1] - p0[1])

    len_a = math.hypot(*ax)
    len_b = math.hypot(*bx)

    ua = (ax[0] / len_a, ax[1] / len_a) if len_a > 1e-9 else (1.0, 0.0)
    ub = (bx[0] / len_b, bx[1] / len_b) if len_b > 1e-9 else (0.0, 1.0)

    cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0
    cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0

    half_a = max(len_a / 2.0, min_size / 2.0)
    half_b = max(len_b / 2.0, min_size / 2.0)

    return [
        (cx - ua[0]*half_a - ub[0]*half_b, cy - ua[1]*half_a - ub[1]*half_b),
        (cx + ua[0]*half_a - ub[0]*half_b, cy + ua[1]*half_a - ub[1]*half_b),
        (cx + ua[0]*half_a + ub[0]*half_b, cy + ua[1]*half_a + ub[1]*half_b),
        (cx - ua[0]*half_a + ub[0]*half_b, cy - ua[1]*half_a + ub[1]*half_b),
    ]


def _normalise_global(
    corners_px: List[Tuple[float, float]],
    img_width: int,
    img_height: int,
) -> List[Tuple[float, float]]:
    """Normalise pixel coordinates to [0, 1] relative to the full image.

    Unlike the old per-tile converter, the denominator here is the full
    processed image dimensions, giving SAHI the correct coordinate space.

    Args:
        corners_px: 4 ``(col, row)`` pixel-coordinate tuples.
        img_width: Full image width in pixels.
        img_height: Full image height in pixels.

    Returns:
        4 ``(x_norm, y_norm)`` tuples in [0, 1].
    """
    return [
        (
            max(0.0, min(1.0, col / img_width)),
            max(0.0, min(1.0, row / img_height)),
        )
        for col, row in corners_px
    ]


def _to_yolo_line(class_id: int, norm: List[Tuple[float, float]]) -> str:
    """Format one YOLO OBB label line.

    Args:
        class_id: Integer YOLO class index.
        norm: 4 ``(x, y)`` normalised coordinate pairs.

    Returns:
        Label string in the form ``"class_id x1 y1 x2 y2 x3 y3 x4 y4"``.
    """
    coords = " ".join(f"{v:.6f}" for pt in norm for v in pt)
    return f"{class_id} {coords}"


# ---------------------------------------------------------------------------
# Public step
# ---------------------------------------------------------------------------

class AnnotationStep(BaseStep):
    """Convert GeoJSON OBBs → YOLO OBB labels normalised to the global image.

    For each processed GeoTIFF in ``cfg["paths"]["spatial_dir"]``, the step:

    1. Finds the matching ``.geojson`` (same stem) in ``raw_dir``.
    2. Reads the image's CRS, Affine, and dimensions directly from the file.
    3. Reprojects annotation corners from WGS 84 → image CRS → pixel space.
    4. Filters by ``min_visible`` against the full image boundary.
    5. Normalises by ``(width, height)`` — not tile size.
    6. Writes one ``.txt`` per image (empty file for background images).
    """

    NAME = "annotations"

    def run(self, cfg: dict) -> None:
        """Execute the global annotation conversion stage.

        Args:
            cfg: Resolved configuration dictionary.  Uses:

                * ``cfg["paths"]["spatial_dir"]`` — stage 2 output.
                * ``cfg["paths"]["raw_dir"]`` — source GeoJSONs.
                * ``cfg["paths"]["labels_dir"]`` — output label directory.
                * ``cfg["annotations"]`` — stage hyperparameters.
        """
        paths  = cfg["paths"]
        params = AnnotationConfig.from_dict(cfg.get("annotations", {}))

        img_dir:    Path = paths["spatial_dir"]
        raw_dir:    Path = paths["raw_dir"]
        labels_dir: Path = paths["labels_dir"]
        labels_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(img_dir.glob("*.tif"))
        if not tif_files:
            raise RuntimeError(f"No .tif files found in '{img_dir}'.")

        logger.info(
            "Annotation conversion: %d image(s)  min_visible=%.2f  min_size_px=%.1f",
            len(tif_files), params.min_visible, params.min_size_px,
        )

        total_annots = total_empty = 0

        for tif_path in tif_files:
            geojson_path = self._find_geojson(raw_dir, tif_path.stem)
            n_written    = self._convert_image(
                tif_path, geojson_path, labels_dir, params
            )
            total_annots += n_written
            if n_written == 0:
                total_empty += 1

        logger.info(
            "Annotation stage complete.  Images=%d  Annotations=%d  "
            "Empty (background)=%d  → '%s'.",
            len(tif_files), total_annots, total_empty, labels_dir,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_geojson(raw_dir: Path, stem: str) -> Optional[Path]:
        """Locate the GeoJSON file matching a processed image stem.

        Args:
            raw_dir: Directory containing raw GeoJSON files.
            stem: Stem of the processed GeoTIFF (e.g. ``"scene_001"``).

        Returns:
            Path to the GeoJSON file, or ``None`` if not found.
        """
        for ext in (".geojson", ".json"):
            candidate = raw_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _convert_image(
        self,
        tif_path: Path,
        geojson_path: Optional[Path],
        labels_dir: Path,
        params: AnnotationConfig,
    ) -> int:
        """Write the YOLO label file for one processed image.

        Args:
            tif_path: Path to the processed GeoTIFF.
            geojson_path: Path to the matching GeoJSON, or ``None``.
            labels_dir: Directory for output ``.txt`` files.
            params: :class:`AnnotationConfig` hyperparameters.

        Returns:
            Number of annotations written (0 for background images).
        """
        label_path = labels_dir / f"{tif_path.stem}.txt"

        if geojson_path is None:
            logger.warning(
                "  No GeoJSON for '%s' — writing empty label.", tif_path.name
            )
            label_path.write_text("")
            return 0

        features = _load_features(geojson_path)

        with rasterio.open(tif_path) as ds:
            crs_str       = str(ds.crs)
            img_transform = ds.transform
            img_width     = ds.width
            img_height    = ds.height

        inv_transform = ~img_transform
        transformer   = _build_transformer("EPSG:4326", crs_str)

        # Build image footprint polygon (in image CRS) for visibility check.
        tl = img_transform * (0,           0)
        tr = img_transform * (img_width,   0)
        br = img_transform * (img_width,   img_height)
        bl = img_transform * (0,           img_height)
        img_poly = Polygon([tl, tr, br, bl])

        lines: List[str] = []

        for feature in features:
            props    = feature.get("properties", {})
            raw_cls  = int(props.get("class_id", -1))

            if raw_cls in params.skip_classes:
                continue

            yolo_cls = params.class_map.get(raw_cls)
            if yolo_cls is None:
                logger.warning(
                    "  Unknown class_id=%d in '%s' — skipping annotation.",
                    raw_cls, geojson_path.name,
                )
                continue

            corners_wgs84 = _obb_corners_wgs84(feature)
            if corners_wgs84 is None:
                continue

            corners_crs = _project_corners(corners_wgs84, transformer)
            ann_poly    = Polygon(corners_crs)

            if not img_poly.intersects(ann_poly):
                continue

            intersection = img_poly.intersection(ann_poly)
            vis_frac = (
                intersection.area / ann_poly.area
                if ann_poly.area > 1e-12 else 0.0
            )
            if vis_frac < params.min_visible:
                continue

            corners_px = _corners_to_pixel(corners_crs, inv_transform)
            corners_px = _enforce_min_side(corners_px, params.min_size_px)
            norm       = _normalise_global(corners_px, img_width, img_height)

            if Polygon(norm).area < 1e-12:
                logger.debug(
                    "  Degenerate polygon in '%s' after normalisation — skipped.",
                    tif_path.name,
                )
                continue

            lines.append(_to_yolo_line(yolo_cls, norm))

        with open(label_path, "w") as fh:
            if lines:
                fh.write("\n".join(lines) + "\n")
            # Empty file = background image — required by YOLO dataloader.

        logger.debug(
            "  %s → %d annotation(s)", tif_path.name, len(lines)
        )
        return len(lines)
