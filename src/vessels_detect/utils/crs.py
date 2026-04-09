"""
src/vessels_detect/utils/crs.py
--------------------------------
CRS (Coordinate Reference System) helpers shared across the pipeline.

All geometry entering or leaving the pipeline must pass through these
helpers so that CRS mismatches are caught and resolved in one place.

The canonical rule for this project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Raw GeoTIFFs are in **EPSG:32739** (WGS 84 / UTM zone 39S) or any
  other projected CRS.
- Mask files (``range.geojson``, ``buildings.geojson``) are assumed to
  be in **EPSG:4326** (WGS-84 geographic).
- Scoring geometry (OBBBox polygons) lives in **EPSG:4326** so that a
  single reference frame is used for spatial filtering and the GeoJSON
  output.

Public API
~~~~~~~~~~
- :func:`get_tif_crs`        - read the CRS string from a GeoTIFF.
- :func:`reproject_geometry` - reproject any Shapely geometry.
- :func:`load_mask_in_tif_crs` - load a GeoJSON mask reprojected to the
                                  image CRS (for pixel-space operations).
- :func:`load_mask_wgs84`    - load a GeoJSON mask and keep it in WGS-84
                                  (for matching against WGS-84 OBBBox
                                  polygons).
- :func:`ensure_same_crs`    - assert / reproject a geometry to a target
                                  CRS, raising clearly if the source CRS is
                                  unknown.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union

import rasterio
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# WGS-84 canonical identifier used throughout the project.
WGS84 = "EPSG:4326"


# =============================================================================
# Low-level helpers
# =============================================================================

def get_tif_crs(tif_path: Path) -> str:
    """Return the CRS of a GeoTIFF as an ``"EPSG:XXXXX"`` string.

    Args:
        tif_path: Path to any readable GeoTIFF.

    Returns:
        CRS string, e.g. ``"EPSG:32739"``.

    Raises:
        ValueError: If the GeoTIFF has no CRS defined.
    """
    with rasterio.open(tif_path) as src:
        crs = src.crs
    if crs is None:
        raise ValueError(f"GeoTIFF has no CRS defined: {tif_path}")
    return crs.to_string()


def _is_wgs84(crs_str: str) -> bool:
    """Return ``True`` when *crs_str* already denotes WGS-84."""
    return crs_str.upper().replace(" ", "") in {"EPSG:4326", "WGS84", "CRS84"}


def reproject_geometry(
    geom,
    src_crs: str,
    dst_crs: str,
):
    """Reproject a Shapely geometry from *src_crs* to *dst_crs*.

    This is the single authoritative reprojection function for the whole
    project.  Every other module that needs reprojection should call this
    rather than instantiating its own ``pyproj.Transformer``.

    Args:
        geom:    Any Shapely geometry (``Polygon``, ``MultiPolygon``, …).
        src_crs: Source CRS string (e.g. ``"EPSG:4326"``).
        dst_crs: Destination CRS string (e.g. ``"EPSG:32739"``).

    Returns:
        Reprojected geometry of the same Shapely type.

    Notes:
        When *src_crs* and *dst_crs* are the same (case-insensitive) the
        geometry is returned unchanged without constructing a transformer.
    """
    if src_crs.upper() == dst_crs.upper():
        return geom

    import pyproj
    from shapely.ops import transform as shapely_transform

    transformer = pyproj.Transformer.from_crs(
        src_crs, dst_crs, always_xy=True
    )
    reprojected = shapely_transform(transformer.transform, geom)

    logger.debug(
        "Reprojected geometry %s → %s  (bounds before: %s, after: %s)",
        src_crs, dst_crs, geom.bounds, reprojected.bounds,
    )
    return reprojected


# =============================================================================
# GeoJSON mask loaders
# =============================================================================

def _load_polygons_from_geojson(geojson_path: Path) -> list[Polygon]:
    """Parse a GeoJSON file and return a list of valid Shapely polygons.

    Args:
        geojson_path: Path to the ``.geojson`` / ``.json`` file.

    Returns:
        List of valid, non-empty Shapely :class:`~shapely.geometry.Polygon`
        objects.  Invalid geometries are repaired with ``.buffer(0)``.

    Raises:
        FileNotFoundError: If *geojson_path* does not exist.
        ValueError: If no valid polygon features are found.
    """
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON mask not found: {geojson_path}")

    with open(geojson_path, encoding="utf-8") as fh:
        gj = json.load(fh)

    polygons: list[Polygon] = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        if geom is None:
            continue
        try:
            poly = shape(geom)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Skipping invalid geometry in '%s': %s",
                geojson_path.name, exc,
            )

    if not polygons:
        raise ValueError(
            f"No valid polygon features found in '{geojson_path}'."
        )

    return polygons


def load_mask_wgs84(geojson_path: Path) -> MultiPolygon:
    """Load a GeoJSON mask and return its union **in WGS-84**.

    Use this when you need to match against :class:`OBBBox` polygons,
    which are always stored in WGS-84 inside the scoring pipeline.

    The file is assumed to already be in EPSG:4326 (the standard for
    GeoJSON per RFC 7946).  If your masks are in a different CRS, call
    :func:`load_mask_in_crs` instead.

    Args:
        geojson_path: Path to a ``.geojson`` file in EPSG:4326.

    Returns:
        :class:`~shapely.geometry.MultiPolygon` union in WGS-84.
    """
    polygons = _load_polygons_from_geojson(geojson_path)
    union    = unary_union(polygons)

    logger.debug(
        "Loaded WGS-84 mask '%s': %d polygon(s), bounds=%s",
        geojson_path.name, len(polygons), union.bounds,
    )
    return union


def load_mask_in_crs(
    geojson_path: Path,
    target_crs: str,
    *,
    geojson_crs: str = WGS84,
) -> MultiPolygon:
    """Load a GeoJSON mask and reproject its union to *target_crs*.

    Use this when you need the mask in the **image CRS** (e.g. for
    pixel-space intersection tests before coordinate conversion).

    Args:
        geojson_path: Path to the ``.geojson`` file.
        target_crs:   Destination CRS string (e.g. ``"EPSG:32739"``).
        geojson_crs:  CRS of the GeoJSON file; defaults to ``"EPSG:4326"``
                      (RFC 7946 compliant).

    Returns:
        :class:`~shapely.geometry.MultiPolygon` union in *target_crs*.
    """
    polygons = _load_polygons_from_geojson(geojson_path)
    union    = unary_union(polygons)

    if not _is_wgs84(target_crs):
        union = reproject_geometry(union, src_crs=geojson_crs, dst_crs=target_crs)

    logger.debug(
        "Loaded mask '%s' reprojected to %s: %d polygon(s), bounds=%s",
        geojson_path.name, target_crs, len(polygons), union.bounds,
    )
    return union


# =============================================================================
# Convenience: derive target CRS from a directory of GeoTIFFs
# =============================================================================

def crs_from_tif_dir(tif_dir: Path) -> Optional[str]:
    """Return the CRS of the first ``*.tif`` found in *tif_dir*.

    This is the standard way to discover the image CRS when all tiles in
    a directory share the same projection (which is always the case inside
    this pipeline).

    Args:
        tif_dir: Directory to search.

    Returns:
        CRS string (e.g. ``"EPSG:32739"``) or ``None`` if no ``.tif``
        files are found.
    """
    first_tif = next(tif_dir.glob("*.tif"), None)
    if first_tif is None:
        logger.warning("No .tif files found in '%s' - cannot determine CRS.", tif_dir)
        return None
    crs = get_tif_crs(first_tif)
    logger.debug("Detected image CRS from '%s': %s", first_tif.name, crs)
    return crs