"""
src/vessels_detect/predict/predictor.py
-----------------------------------------
YOLO inference on a directory of GeoTIFF tiles, followed by per-image
GeoJSON assembly.

Responsibilities
~~~~~~~~~~~~~~~~
1.  Load the YOLO-OBB model from the checkpoint path in the config.
2.  Iterate over every ``*.tif`` in ``tiles_dir``.
3.  Run the model on each tile and collect raw detections.
4.  Apply global cross-tile NMS to remove duplicate detections on tile
    boundaries.
5.  Group detections by *source image* (derived from the tile filename
    stem - see :func:`_source_stem_from_tile`).
6.  Write one ``<image_stem>.geojson`` per source image to ``predictions_dir``.

Tile filename convention
~~~~~~~~~~~~~~~~~~~~~~~~
Tiles are expected to follow the convention produced by the tiling
preprocessing step::

    <image_stem>_<x_off>_<y_off>.tif

The ``x_off`` and ``y_off`` components are non-negative integers.
:func:`_source_stem_from_tile` strips them to recover ``<image_stem>``,
which is then used to group tiles and to look up the source GeoTIFF's
Affine transform and CRS.

Coordinate pipeline (per detection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    YOLO prediction  (pixel coords relative to the tile image)
        ↓  normalise by tile dimensions
    Normalised tile coords  [0, 1]
        ↓  tile_obb_to_image_pixels  (+ tile x_off, y_off)
    Absolute pixel coords in the source image
        ↓  pixels_to_crs  (Affine transform of the source GeoTIFF)
    Image CRS  (e.g. EPSG:32631 UTM)
        ↓  crs_to_wgs84
    WGS-84  →  GeoJSON feature

The coordinate helpers are imported from
:mod:`src.vessels_detect.postprocessing.geojson_writer` which is the
single source of truth for the coordinate pipeline in this project.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from src.vessels_detect.postprocessing.geojson_writer import (
    parse_ultralytics_obb_results,
    write_prediction_geojson,
)
from src.vessels_detect.postprocessing.spatial_filter import OBBBox

logger = logging.getLogger(__name__)

# Tile stem suffix pattern: trailing _<digits>_<digits>
_TILE_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _source_stem_from_tile(tile_stem: str) -> Tuple[str, int, int]:
    """Extract source image stem and tile offsets from a tile filename stem.

    Args:
        tile_stem: Tile filename without extension,
                   e.g. ``"image_a_0_640"``.

    Returns:
        ``(source_stem, x_off, y_off)`` where ``source_stem`` is
        ``"image_a"`` for the example above.

    Raises:
        ValueError: If the stem does not end with ``_<int>_<int>``.
    """
    m = _TILE_SUFFIX_RE.search(tile_stem)
    if m is None:
        raise ValueError(
            f"Tile stem '{tile_stem}' does not match expected pattern "
            f"'<image_stem>_<x_off>_<y_off>'."
        )
    x_off = int(m.group(1))
    y_off = int(m.group(2))
    source_stem = tile_stem[: m.start()]
    return source_stem, x_off, y_off


def _global_nms(
    boxes: List[OBBBox],
    iou_threshold: float,
) -> List[OBBBox]:
    """Greedy IoU-based NMS across all tiles for a single source image.

    Detections are sorted by descending confidence.  A detection is
    suppressed when its IoU with any already-kept detection meets or
    exceeds *iou_threshold*.

    Args:
        boxes:         All detections for one source image (any number of
                       tiles), already in WGS-84.
        iou_threshold: IoU threshold above which two boxes are considered
                       duplicates.

    Returns:
        Filtered list of detections; original ``OBBBox`` instances are
        reused (not copied).
    """
    if not boxes:
        return []

    # Sort descending by confidence (NaN → 0).
    sorted_boxes = sorted(
        boxes, key=lambda b: b.confidence if b.confidence == b.confidence else 0.0,
        reverse=True,
    )

    kept: List[OBBBox] = []
    for candidate in sorted_boxes:
        suppress = False
        for retained in kept:
            if candidate.class_id != retained.class_id:
                continue
            intersection = candidate.polygon.intersection(retained.polygon).area
            union = candidate.polygon.union(retained.polygon).area
            if union > 0 and intersection / union >= iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)

    logger.debug(
        "Global NMS: %d → %d detections (threshold=%.2f).",
        len(boxes), len(kept), iou_threshold,
    )
    return kept


def _read_tile_meta(tile_path: Path) -> Tuple[rasterio.Affine, str, int, int]:
    """Read the Affine transform, CRS, width, and height of a GeoTIFF tile.

    Args:
        tile_path: Path to the tile GeoTIFF.

    Returns:
        ``(affine, crs_string, width, height)``
    """
    with rasterio.open(tile_path) as src:
        return src.transform, src.crs.to_string(), src.width, src.height


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Runs YOLO inference on tiled GeoTIFFs and writes per-image GeoJSONs.

    Instantiated by :class:`~src.vessels_detect.manager.PredictManager`.
    Stateless across ``run()`` calls.
    """

    def run(self, cfg: dict) -> int:
        """Execute inference for all tiles and write raw prediction GeoJSONs.

        Args:
            cfg: Fully resolved config dict from
                 :func:`~src.vessels_detect.manager.load_config`.

        Returns:
            Exit code: ``0`` on success, ``1`` on fatal error.
        """
        try:
            return self._run(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.critical("Predictor failed: %s", exc, exc_info=True)
            return 1

    def _run(self, cfg: dict) -> int:
        tiles_dir:       Path  = cfg["paths"]["tiles_dir"]
        predictions_dir: Path  = cfg["paths"]["predictions_dir"]
        weights:         str   = cfg["model"]["weights"]
        class_names:     Dict  = {int(k): v for k, v in cfg["model"]["class_names"].items()}
        pred_cfg:        dict  = cfg["prediction"]

        predictions_dir.mkdir(parents=True, exist_ok=True)

        tile_paths = sorted(tiles_dir.glob("*.tif"))
        if not tile_paths:
            logger.error("No .tif files found in '%s'.", tiles_dir)
            return 1

        logger.info("Loading model from '%s'.", weights)
        model = self._load_model(weights, pred_cfg)

        # Group tiles by source image stem.
        groups: Dict[str, List[Path]] = defaultdict(list)
        for tp in tile_paths:
            try:
                source_stem, _, _ = _source_stem_from_tile(tp.stem)
            except ValueError as exc:
                logger.warning("%s - skipping tile.", exc)
                continue
            groups[source_stem].append(tp)

        logger.info(
            "Running inference on %d tile(s) from %d source image(s).",
            len(tile_paths), len(groups),
        )

        for source_stem, tiles in groups.items():
            logger.info("Processing source image: %s  (%d tile(s))", source_stem, len(tiles))
            all_detections = self._predict_tiles(
                model, tiles, source_stem, class_names, pred_cfg
            )

            # Global NMS across overlapping tiles.
            all_detections = _global_nms(all_detections, pred_cfg["global_nms_iou"])

            out_path = predictions_dir / f"{source_stem}.geojson"
            write_prediction_geojson(
                all_detections,
                out_path,
                indent=pred_cfg.get("geojson_indent", 2),
            )

        logger.info("Predictions written to '%s'.", predictions_dir)
        return 0

    @staticmethod
    def _load_model(weights: str, pred_cfg: dict):
        """Load the Ultralytics YOLO-OBB model.

        Args:
            weights:  Path to the ``.pt`` checkpoint.
            pred_cfg: ``cfg["prediction"]`` sub-dict.

        Returns:
            Loaded :class:`ultralytics.YOLO` model.
        """
        from ultralytics import YOLO  # deferred - heavy import

        model = YOLO(weights)
        return model

    @staticmethod
    def _predict_tiles(
        model,
        tiles: List[Path],
        source_stem: str,
        class_names: Dict[int, str],
        pred_cfg: dict,
    ) -> List[OBBBox]:
        """Run inference on all tiles belonging to one source image.

        Args:
            model:       Loaded Ultralytics YOLO model.
            tiles:       Tile paths for this source image.
            source_stem: Source image stem (used in ``OBBBox.source_image``).
            class_names: ``{class_id: name}`` mapping.
            pred_cfg:    ``cfg["prediction"]`` sub-dict.

        Returns:
            Flat list of :class:`OBBBox` detections in WGS-84.
        """
        all_boxes: List[OBBBox] = []

        for tile_path in tiles:
            _, x_off, y_off = _source_stem_from_tile(tile_path.stem)
            affine, crs_str, tile_w, tile_h = _read_tile_meta(tile_path)

            results = model.predict(
                source=str(tile_path),
                conf=pred_cfg["conf"],
                iou=pred_cfg["iou"],
                imgsz=pred_cfg["imgsz"],
                verbose=False,
            )

            if not results:
                continue

            boxes = parse_ultralytics_obb_results(
                results   = results[0],
                tile_path = tile_path,
                source_image_stem = source_stem,
                x_off     = 0,
                y_off     = 0,
                tile_size = tile_w,  # assumes square tiles; edge tiles same size (padded)
                affine    = affine,
                src_crs   = crs_str,
                class_names = class_names,
            )
            all_boxes.extend(boxes)

        return all_boxes
