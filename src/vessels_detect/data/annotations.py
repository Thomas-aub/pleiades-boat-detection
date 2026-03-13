"""
src/vessels_detect/data/annotations.py
------------------------
Converts raw GeoJSON oriented bounding-box (OBB) boat annotations into
YOLO OBB ``.txt`` label files, one file per GeoTIFF tile.

The spatial pipeline is entirely self-contained: each tile's CRS and
Affine transform are read directly from the GeoTIFF produced by
:mod:`src.data.tiler`. No external metadata file is required.

Coordinate flow per annotation:
    1. GeoJSON exterior ring (WGS 84, EPSG:4326) →
    2. Minimum rotated rectangle (Shapely OBB) →
    3. Project to tile's native CRS with pyproj →
    4. Apply inverse tile Affine → pixel-space (col, row) →
    5. Enforce minimum side-length →
    6. Normalise to [0, 1] relative to ``tile_size`` →
    7. Write YOLO OBB format:  ``class_id x1 y1 x2 y2 x3 y3 x4 y4``

One ``.txt`` is written for every tile even when it contains no
annotations (empty file = background tile, required by YOLO).

Typical usage::

    from src.data.annotations import AnnotationConverter, AnnotationConfig

    config = AnnotationConfig(tile_size=320, min_visible=0.25)
    converter = AnnotationConverter(config)
    converter.convert_directory(
        tiles_dir=Path("data/processed/tiles"),
        raw_dir=Path("data/raw"),
        labels_dir=Path("data/processed/labels_raw"),
    )
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import pyproj
import rasterio
from affine import Affine
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnnotationConfig:
    """Hyperparameters for :class:`AnnotationConverter`.

    Attributes:
        tile_size: Padded tile size in pixels (must match tiling step).
            Used as the denominator when normalising pixel coordinates
            to the [0, 1] range expected by YOLO OBB.
        min_visible: Minimum fraction of an OBB's area that must fall
            inside the tile for the annotation to be included.  OBBs
            with a smaller visible fraction are silently dropped.
        min_size_border: Minimum side length in pixels for any OBB axis.
            Boxes smaller than this threshold are symmetrically elongated
            from their geometric centre to reach ``min_size_border``.
        class_map: Mapping from GeoJSON ``class_id`` integers to YOLO
            class indices.  Classes absent from both ``class_map`` and
            ``skip_classes`` trigger a warning.
        skip_classes: Set of GeoJSON ``class_id`` values to discard
            entirely; no label is written for these annotations.
    """

    tile_size: int = 320
    min_visible: float = 0.25
    min_size_border: float = 1.0
    class_map: Dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4}
    )
    skip_classes: Set[int] = field(default_factory=lambda: {9})

    @classmethod
    def from_dict(cls, cfg: dict) -> "AnnotationConfig":
        """Construct from a plain dictionary (e.g., parsed YAML section).

        Args:
            cfg: Dictionary with keys matching the dataclass field names.
                ``class_map`` keys and values are coerced to ``int``.

        Returns:
            A populated :class:`AnnotationConfig` instance.
        """
        raw_class_map = cfg.get("class_map", {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4})
        return cls(
            tile_size=cfg.get("tile_size", 320),
            min_visible=cfg.get("min_visible", 0.25),
            min_size_border=cfg.get("min_size_border", 1.0),
            class_map={int(k): int(v) for k, v in raw_class_map.items()},
            skip_classes=set(cfg.get("skip_classes", [9])),
        )


# ---------------------------------------------------------------------------
# GeoJSON + geometry helpers
# ---------------------------------------------------------------------------

def _load_features(geojson_path: Path) -> List[dict]:
    """Parse a GeoJSON file and return its feature list.

    Args:
        geojson_path: Path to a ``.geojson`` or ``.json`` file.

    Returns:
        List of GeoJSON feature dictionaries.  Returns an empty list if
        the file contains no ``"features"`` key.

    Raises:
        OSError: If the file cannot be opened.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(geojson_path) as fh:
        gj = json.load(fh)
    return gj.get("features", [])


def _obb_from_feature(feature: dict) -> Optional[List[Tuple[float, float]]]:
    """Extract exactly 4 corner coordinates from a GeoJSON polygon feature.

    If the input polygon has more or fewer than 4 vertices, Shapely's
    ``minimum_rotated_rectangle`` forces it into a valid oriented bounding
    box.  Self-intersecting rings are repaired with ``buffer(0)`` before
    computing the OBB.

    Args:
        feature: A GeoJSON Feature dictionary with a ``Polygon`` geometry.

    Returns:
        List of 4 ``(lon, lat)`` tuples (WGS 84) forming the OBB, or
        ``None`` if the geometry is absent, malformed, or unfixable.
    """
    try:
        coords = feature["geometry"]["coordinates"][0]  # outer ring
        poly   = Polygon(coords)

        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                return None

        obb    = poly.minimum_rotated_rectangle
        pts    = list(obb.exterior.coords)
        # GeoJSON closed rings repeat the first vertex at the end.
        if pts[0] == pts[-1]:
            pts = pts[:-1]

        if len(pts) != 4:
            logger.debug("OBB produced %d points (expected 4) — skipping.", len(pts))
            return None

        return [(float(p[0]), float(p[1])) for p in pts]

    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to extract OBB from feature: %s", exc)
        return None


def _enforce_min_side(
    corners_px: List[Tuple[float, float]],
    min_size: float,
) -> List[Tuple[float, float]]:
    """Symmetrically elongate OBB axes that are shorter than *min_size*.

    The OBB is parameterised by its centre and two half-axis vectors.
    If either axis length is below ``min_size``, it is extended to
    ``min_size`` while the orthogonal axis and the box centre are kept
    fixed.

    Args:
        corners_px: Four ``(x, y)`` tuples in pixel coordinates, ordered
            so that consecutive vertices share an edge.
        min_size: Minimum allowable side length in pixels.

    Returns:
        Four ``(x, y)`` tuples, potentially adjusted.
    """
    p0, p1, p2, p3 = corners_px

    # Axis vectors along each pair of opposite sides.
    ax = (p1[0] - p0[0], p1[1] - p0[1])
    bx = (p3[0] - p0[0], p3[1] - p0[1])

    len_a = math.hypot(*ax)
    len_b = math.hypot(*bx)

    # Unit vectors (guard against degenerate zero-length sides).
    ua = (ax[0] / len_a, ax[1] / len_a) if len_a > 1e-9 else (1.0, 0.0)
    ub = (bx[0] / len_b, bx[1] / len_b) if len_b > 1e-9 else (0.0, 1.0)

    cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0
    cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0

    half_a = max(len_a / 2.0, min_size / 2.0)
    half_b = max(len_b / 2.0, min_size / 2.0)

    new_p0 = (cx - ua[0] * half_a - ub[0] * half_b, cy - ua[1] * half_a - ub[1] * half_b)
    new_p1 = (cx + ua[0] * half_a - ub[0] * half_b, cy + ua[1] * half_a - ub[1] * half_b)
    new_p2 = (cx + ua[0] * half_a + ub[0] * half_b, cy + ua[1] * half_a + ub[1] * half_b)
    new_p3 = (cx - ua[0] * half_a + ub[0] * half_b, cy - ua[1] * half_a + ub[1] * half_b)

    return [new_p0, new_p1, new_p2, new_p3]


def _normalise(corners_px: List[Tuple[float, float]], tile_size: int) -> List[Tuple[float, float]]:
    """Normalise pixel coordinates to the [0, 1] range.

    Args:
        corners_px: Four ``(x, y)`` pixel-space tuples.
        tile_size: Tile edge length in pixels (same for width and height).

    Returns:
        Four ``(x_norm, y_norm)`` tuples, each in [0, 1].
    """
    return [(x / tile_size, y / tile_size) for x, y in corners_px]


def _to_yolo_line(class_id: int, corners: List[Tuple[float, float]]) -> str:
    """Serialise one OBB to a YOLO OBB label line.

    YOLO OBB format::

        class_id  x1 y1  x2 y2  x3 y3  x4 y4

    All coordinates are normalised fractions in [0, 1].

    Args:
        class_id: Integer YOLO class index.
        corners: Four ``(x_norm, y_norm)`` tuples.

    Returns:
        A single-line string (no trailing newline).
    """
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in corners)
    return f"{class_id} {coords}"


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _build_transformer(
    src_crs: str,
    dst_crs_str: str,
) -> Optional[pyproj.Transformer]:
    """Build a pyproj coordinate transformer from WGS 84 to *dst_crs_str*.

    Args:
        src_crs: Source CRS identifier (always ``"EPSG:4326"`` for GeoJSON).
        dst_crs_str: Destination CRS as a string (e.g., ``"EPSG:32632"``),
            as returned by :attr:`rasterio.DatasetReader.crs`.

    Returns:
        A :class:`pyproj.Transformer` instance, or ``None`` if the
        destination CRS cannot be parsed (in which case a warning is logged
        and the caller should treat coordinates as already in the dst CRS).
    """
    try:
        return pyproj.Transformer.from_crs(src_crs, dst_crs_str, always_xy=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not build transformer to CRS '%s': %s. "
            "GeoJSON coordinates will be used as-is.",
            dst_crs_str, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Per-image annotation processing
# ---------------------------------------------------------------------------

def _prepare_annotations(
    features: List[dict],
    transformer: Optional[pyproj.Transformer],
    class_map: Dict[int, int],
    skip_classes: Set[int],
) -> List[Tuple[int, List[Tuple[float, float]], Polygon]]:
    """Filter, project, and pre-build Shapely polygons for all features.

    This is called once per source image. The resulting list is reused
    across all tiles derived from that image, avoiding repeated projection
    and polygon construction.

    Args:
        features: Raw GeoJSON feature dictionaries.
        transformer: pyproj transformer from EPSG:4326 to tile CRS,
            or ``None`` to skip projection.
        class_map: ``{geojson_class_id: yolo_class_id}`` mapping.
        skip_classes: Set of GeoJSON class IDs to discard.

    Returns:
        List of ``(yolo_class_id, projected_corners, projected_polygon)``
        tuples for annotations that passed all filters.
    """
    prepared: List[Tuple[int, List[Tuple[float, float]], Polygon]] = []
    n_skip_cls = n_bad_geom = n_bad_proj = 0

    for feat in features:
        cid = feat.get("properties", {}).get("class_id")

        # Class filtering.
        if cid in skip_classes:
            n_skip_cls += 1
            continue
        if cid not in class_map:
            logger.warning("Unknown class_id %s — skipping annotation.", cid)
            n_skip_cls += 1
            continue

        # Geometry extraction.
        corners_wgs84 = _obb_from_feature(feat)
        if corners_wgs84 is None:
            n_bad_geom += 1
            continue

        # Coordinate projection.
        try:
            if transformer is not None:
                corners_proj = [
                    transformer.transform(lon, lat) for lon, lat in corners_wgs84
                ]
            else:
                corners_proj = corners_wgs84

            poly = Polygon(corners_proj)
            if not poly.is_valid:
                poly = poly.buffer(0)

        except Exception as exc:  # noqa: BLE001
            logger.debug("Projection failed for annotation: %s", exc)
            n_bad_proj += 1
            continue

        prepared.append((class_map[int(cid)], corners_proj, poly))

    logger.debug(
        "  Prepared %d annotation(s) | skip_cls=%d bad_geom=%d bad_proj=%d",
        len(prepared), n_skip_cls, n_bad_geom, n_bad_proj,
    )
    return prepared


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AnnotationConverter:
    """Generates YOLO OBB ``.txt`` label files from GeoJSON annotations.

    For each GeoTIFF tile in *tiles_dir*, the converter:
        1. Reads the tile's CRS and Affine transform directly from the file.
        2. Locates the corresponding GeoJSON (matched by source filename
           stored in the tile's TIFF tags).
        3. Filters, projects, and clips annotations to the tile footprint.
        4. Writes a YOLO OBB ``.txt`` file (empty for background tiles).

    Args:
        config: :class:`AnnotationConfig` with all hyperparameters.
    """

    def __init__(self, config: AnnotationConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def convert_directory(
        self,
        tiles_dir: Path,
        raw_dir: Path,
        labels_dir: Path,
    ) -> None:
        """Generate label files for every tile in *tiles_dir*.

        Tiles are grouped by their ``source_tif`` TIFF tag so that the
        corresponding GeoJSON is loaded and its annotations projected only
        once per source image — not once per tile.

        Args:
            tiles_dir: Directory containing GeoTIFF tiles (step 1 output).
            raw_dir: Directory containing the original ``.geojson`` files.
                Files must share the stem of their corresponding ``.tif``
                (e.g., ``scene_001.geojson`` pairs with ``scene_001.tif``).
            labels_dir: Directory where ``.txt`` label files will be
                written.  Created automatically if absent.

        Raises:
            FileNotFoundError: If *tiles_dir* does not exist.
            RuntimeError: If no ``.tif`` tiles are found in *tiles_dir*.
        """
        if not tiles_dir.exists():
            raise FileNotFoundError(f"tiles_dir does not exist: {tiles_dir}")

        tile_paths = sorted(tiles_dir.glob("*.tif"))
        if not tile_paths:
            raise RuntimeError(f"No .tif tiles found in '{tiles_dir}'.")

        labels_dir.mkdir(parents=True, exist_ok=True)

        # Group tile paths by their source image (read from TIFF tag).
        by_source: Dict[str, List[Path]] = defaultdict(list)
        for tp in tile_paths:
            try:
                with rasterio.open(tp) as ds:
                    source = ds.tags().get("source_tif", tp.stem)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cannot read tags from '%s': %s — skipping.", tp.name, exc)
                continue
            by_source[source].append(tp)

        logger.info(
            "Converting annotations for %d tile(s) from %d source image(s).",
            len(tile_paths), len(by_source),
        )

        total_labels = total_annots = total_empty = 0

        for source_name, group in by_source.items():
            tif_stem     = Path(source_name).stem
            geojson_path = raw_dir / f"{tif_stem}.geojson"

            if not geojson_path.exists():
                # Search for alternative extension.
                alt = raw_dir / f"{tif_stem}.json"
                if alt.exists():
                    geojson_path = alt
                else:
                    logger.warning(
                        "No GeoJSON found for '%s' — all %d tile(s) will be "
                        "written as empty (background) labels.",
                        source_name, len(group),
                    )
                    features = []

            if geojson_path.exists():
                features = _load_features(geojson_path)
                logger.info(
                    "  %s  →  %s  (%d raw annotation(s), %d tile(s))",
                    source_name, geojson_path.name, len(features), len(group),
                )

            # Read CRS from the first tile in the group (all tiles from the
            # same source share the same CRS).
            crs_str: Optional[str] = None
            try:
                with rasterio.open(group[0]) as ds:
                    crs_str = str(ds.crs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cannot read CRS from '%s': %s", group[0].name, exc)

            transformer = _build_transformer("EPSG:4326", crs_str) if crs_str else None

            # Pre-process annotations once for the entire source image.
            prepared = _prepare_annotations(
                features, transformer,
                self._cfg.class_map, self._cfg.skip_classes,
            )

            # Process each tile.
            for tile_path in group:
                n_written = self._convert_tile(tile_path, prepared, labels_dir)
                total_labels += 1
                total_annots += n_written
                if n_written == 0:
                    total_empty += 1

        logger.info(
            "Annotation conversion complete. "
            "Files: %d | Annotations: %d | Empty (background): %d",
            total_labels, total_annots, total_empty,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _convert_tile(
        self,
        tile_path: Path,
        prepared: List[Tuple[int, List[Tuple[float, float]], Polygon]],
        labels_dir: Path,
    ) -> int:
        """Write the YOLO label file for a single tile.

        Args:
            tile_path: Path to the GeoTIFF tile.
            prepared: Pre-processed annotation list from
                :func:`_prepare_annotations`.
            labels_dir: Directory where the ``.txt`` file will be written.

        Returns:
            Number of annotations written to the label file (0 for
            background tiles).
        """
        cfg = self._cfg

        try:
            with rasterio.open(tile_path) as ds:
                tile_transform = ds.transform
                tile_w         = ds.width
                tile_h         = ds.height
        except Exception as exc:  # noqa: BLE001
            logger.error("Cannot open tile '%s': %s — skipping.", tile_path.name, exc)
            return 0

        inv_transform = ~tile_transform

        # Geographic footprint of this tile (in the tile's native CRS).
        tl = tile_transform * (0,      0)
        tr = tile_transform * (tile_w, 0)
        br = tile_transform * (tile_w, tile_h)
        bl = tile_transform * (0,      tile_h)
        tile_poly = Polygon([tl, tr, br, bl])

        lines: List[str] = []

        for yolo_cls, corners_proj, ann_poly in prepared:
            if not tile_poly.intersects(ann_poly):
                continue

            intersection = tile_poly.intersection(ann_poly)
            if intersection.is_empty:
                continue

            vis_frac = (
                intersection.area / ann_poly.area
                if ann_poly.area > 1e-12 else 0.0
            )
            if vis_frac < cfg.min_visible:
                continue

            # Map coordinates → pixel coordinates (relative to tile origin).
            corners_px = [inv_transform * (x, y) for x, y in corners_proj]

            # Enforce minimum OBB side length.
            corners_px = _enforce_min_side(corners_px, cfg.min_size_border)

            norm = _normalise(corners_px, cfg.tile_size)

            # Skip degenerate polygons after normalisation.
            if Polygon(norm).area < 1e-12:
                logger.debug("Degenerate normalised polygon in '%s' — skipped.", tile_path.name)
                continue

            lines.append(_to_yolo_line(yolo_cls, norm))

        label_path = labels_dir / f"{tile_path.stem}.txt"
        with open(label_path, "w") as fh:
            if lines:
                fh.write("\n".join(lines))
            # Empty file for background tiles — YOLO requires the file to exist.

        return len(lines)
