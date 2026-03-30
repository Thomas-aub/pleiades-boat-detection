"""
src/inference/predictor.py
---------------------------
Runs YOLO-OBB inference over a directory of GeoTIFF tiles and yields
structured, tile-level prediction records.

Each record bundles the raw pixel-space bounding-box output from YOLO
together with the tile's full geospatial context (path, CRS, Affine
transform) read directly from the GeoTIFF file.  Downstream modules
(:mod:`src.inference.postprocess` and :mod:`src.inference.geospatial`)
consume these records without ever re-opening the images.

Why GeoTIFF instead of PNG?
    The tiling step (:mod:`src.data.tiler`) now writes fully georeferenced
    GeoTIFF files.  Reading the CRS and Affine transform from the same file
    that YOLO predicts on removes any dependency on an external metadata
    table and guarantees that pixel coordinates and geospatial context are
    always in sync.

Typical usage::

    from src.utils.config import load_config
    from src.inference.predictor import YoloPredictor

    cfg       = load_config("configs/inference.yaml")
    predictor = YoloPredictor(cfg)
    for record in predictor.predict_directory(Path("data/processed/images/test")):
        # record.pixel_corners  : (N, 4, 2) float32 array of OBB corners
        # record.class_ids      : (N,)      int array
        # record.confidences    : (N,)      float array
        # record.tile_transform : rasterio Affine
        # record.crs            : str (e.g. "EPSG:32632")
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine

from src.vessels_detect.utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TilePrediction:
    """Container for raw YOLO-OBB predictions on a single GeoTIFF tile.

    All arrays share the same length ``N`` (number of detections on this
    tile).  When ``N == 0`` the tile produced no detections above the
    configured confidence threshold.

    Attributes:
        tile_path: Absolute path to the source GeoTIFF tile.
        pixel_corners: OBB corner coordinates in pixel space.
            Shape ``(N, 4, 2)`` — four ``(col, row)`` pairs per detection,
            ordered as returned by ``results.obb.xyxyxyxy``.
        class_ids: Predicted class indices.  Shape ``(N,)``, dtype ``int``.
        confidences: Predicted confidence scores.  Shape ``(N,)``; values
            in ``[0, 1]``.
        tile_transform: Affine transform of the source tile (pixel → CRS).
        crs: CRS string of the source tile (e.g. ``"EPSG:32632"``).
        tile_width: Tile width in pixels (from the GeoTIFF profile).
        tile_height: Tile height in pixels (from the GeoTIFF profile).
    """

    tile_path:       Path
    pixel_corners:   np.ndarray          # (N, 4, 2)  float32
    class_ids:       np.ndarray          # (N,)        int
    confidences:     np.ndarray          # (N,)        float32
    tile_transform:  Affine
    crs:             str
    tile_width:      int
    tile_height:     int

    @property
    def n_detections(self) -> int:
        """Number of detections on this tile."""
        return int(self.class_ids.shape[0])

    @property
    def has_detections(self) -> bool:
        """``True`` if at least one detection was found."""
        return self.n_detections > 0


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class YoloPredictor:
    """Runs YOLO-OBB inference over a directory of GeoTIFF tiles.

    The model is loaded once at construction time.  Inference is performed
    image-by-image (no batching) so that geospatial metadata from the
    GeoTIFF can be attached to each prediction without complicating the
    batch-index bookkeeping.

    Args:
        config: :class:`~src.utils.config.Config` loaded from
            ``configs/inference.yaml``.

    Raises:
        ImportError: If ``ultralytics`` is not installed.
        FileNotFoundError: If the configured model weights file does not
            exist.
    """

    def __init__(self, config: Config) -> None:
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'ultralytics' package is required. "
                "Install with:  pip install ultralytics"
            ) from exc

        self._cfg = config
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_directory(
        self,
        tiles_dir: Path,
    ) -> Generator[TilePrediction, None, None]:
        """Yield one :class:`TilePrediction` per GeoTIFF tile in *tiles_dir*.

        Tiles are processed in sorted order for reproducibility.  Files
        with extensions ``.tif`` or ``.tiff`` are processed; others are
        silently ignored.

        Args:
            tiles_dir: Directory containing GeoTIFF tiles.

        Yields:
            A :class:`TilePrediction` for every tile, including tiles with
            zero detections (useful for computing recall-based metrics).

        Raises:
            FileNotFoundError: If *tiles_dir* does not exist.
            RuntimeError: If no ``.tif`` / ``.tiff`` files are found.
        """
        if not tiles_dir.exists():
            raise FileNotFoundError(f"tiles_dir not found: {tiles_dir}")

        tile_paths = sorted(
            p for p in tiles_dir.iterdir()
            if p.suffix.lower() in {".tif", ".tiff"}
        )
        if not tile_paths:
            raise RuntimeError(f"No GeoTIFF tiles found in '{tiles_dir}'.")

        inf_cfg = self._cfg.inference
        conf    = float(inf_cfg.get("conf", 0.10))
        iou     = float(inf_cfg.get("iou", 0.30))
        imgsz   = int(inf_cfg.get("imgsz", 640))

        logger.info(
            "Predicting on %d tile(s) from '%s'  [conf=%.2f  iou=%.2f  imgsz=%d]",
            len(tile_paths), tiles_dir, conf, iou, imgsz,
        )

        n_with_dets = 0
        total_dets  = 0

        for idx, tile_path in enumerate(tile_paths, 1):
            record = self._predict_tile(tile_path, conf, iou, imgsz)

            if record is None:
                continue  # unreadable tile; already logged

            n_with_dets += int(record.has_detections)
            total_dets  += record.n_detections

            if idx % 100 == 0 or idx == len(tile_paths):
                logger.info(
                    "  %d / %d tiles processed  |  %d detections so far.",
                    idx, len(tile_paths), total_dets,
                )

            yield record

        logger.info(
            "Prediction complete.  Tiles with detections: %d / %d  |  "
            "Total raw detections: %d",
            n_with_dets, len(tile_paths), total_dets,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load YOLO weights and return the model instance.

        Returns:
            An ``ultralytics.YOLO`` model.

        Raises:
            FileNotFoundError: If the weights file does not exist.
        """
        from ultralytics import YOLO

        weights = Path(str(self._cfg.model.weights))
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found: '{weights}'. "
                "Train a model first with scripts/train.py or provide a "
                "pretrained checkpoint."
            )

        logger.info("Loading YOLO model from '%s'.", weights)
        model = YOLO(str(weights))
        logger.info("Model loaded.  Classes: %s", model.names)
        return model

    def _predict_tile(
        self,
        tile_path: Path,
        conf: float,
        iou: float,
        imgsz: int,
    ) -> Optional[TilePrediction]:
        """Run inference on a single tile and return a :class:`TilePrediction`.

        Args:
            tile_path: Path to the GeoTIFF tile.
            conf: Confidence threshold (detections below are suppressed).
            iou: NMS IoU threshold used by YOLO's internal NMS pass.
            imgsz: Inference image size in pixels.

        Returns:
            A :class:`TilePrediction` instance, or ``None`` if the tile
            could not be opened or YOLO encountered an error.
        """
        # ── 1. Read geospatial metadata from the GeoTIFF ──────────────
        try:
            with rasterio.open(tile_path) as ds:
                tile_transform = ds.transform
                crs_str        = str(ds.crs)
                tile_w         = ds.width
                tile_h         = ds.height
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cannot open GeoTIFF '%s': %s — skipping.", tile_path.name, exc
            )
            return None

        # ── 2. YOLO inference ──────────────────────────────────────────
        try:
            results = self._model.predict(
                source=str(tile_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "YOLO prediction failed on '%s': %s — skipping.",
                tile_path.name, exc,
            )
            return None

        # ── 3. Extract OBB tensors ─────────────────────────────────────
        if results.obb is not None and len(results.obb) > 0:
            # xyxyxyxy: (N, 4, 2) tensor of corner pixel coordinates
            pixel_corners = results.obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2)
            class_ids     = results.obb.cls.cpu().numpy().astype(int)
            confidences   = results.obb.conf.cpu().numpy().astype(np.float32)
        else:
            pixel_corners = np.empty((0, 4, 2), dtype=np.float32)
            class_ids     = np.empty((0,),      dtype=int)
            confidences   = np.empty((0,),      dtype=np.float32)

        logger.debug(
            "  %s  →  %d detection(s).", tile_path.name, class_ids.shape[0]
        )

        return TilePrediction(
            tile_path=tile_path,
            pixel_corners=pixel_corners,
            class_ids=class_ids,
            confidences=confidences,
            tile_transform=tile_transform,
            crs=crs_str,
            tile_width=tile_w,
            tile_height=tile_h,
        )

    @property
    def class_names(self) -> dict:
        """Return the ``{class_id: class_name}`` mapping from the model.

        Returns:
            The ``model.names`` dictionary.
        """
        return self._model.names
