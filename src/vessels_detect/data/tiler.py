"""
src/vessels_detect/data/tiler.py
-----------------
Tiles large GeoTIFF satellite images into smaller square GeoTIFF tiles.

Each output tile is a self-contained, valid GeoTIFF that embeds:
  - Its own Coordinate Reference System (CRS)
  - Its own Affine Transform (pixel-space → map-space mapping)
  - Provenance tags: source filename, pixel offsets, original dimensions

This design eliminates any external lookup table (e.g., ``metadata.csv``).
Downstream steps — annotation generation, inference, and prediction
reconstruction — can derive all spatial context by opening the tile alone.

Typical usage::

    from src.data.tiler import ImageTiler, TilerConfig

    config = TilerConfig(tile_size=320, overlap=64)
    tiler  = ImageTiler(config)
    paths  = tiler.tile_directory(
        raw_dir=Path("data/raw"),
        output_dir=Path("data/processed/tiles"),
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TilerConfig:
    """Hyperparameters for :class:`ImageTiler`.

    Attributes:
        tile_size: Output tile width and height in pixels. All tiles are
            square; edge tiles are zero-padded to this size.
        overlap: Pixel overlap between adjacent tiles in both axes.
            Set to 0 for non-overlapping tiles.
        min_percentile: Lower percentile used for per-image global
            contrast stretching (applied once per source image so all
            derived tiles share a consistent colour rendering).
        max_percentile: Upper percentile for contrast stretching.
        bands: 1-based band indices to use as (R, G, B). ``None`` enables
            auto-detection: for images with ≥3 bands the first three are
            used; for single-band (panchromatic) images the band is
            replicated into all three channels.
        compress: GeoTIFF compression codec for output tiles.
            ``"lzw"`` (lossless, fast decode) is recommended for
            byte-range uint8 imagery.
    """

    tile_size: int = 320
    overlap: int = 64
    min_percentile: float = 1.0
    max_percentile: float = 99.0
    bands: Optional[List[int]] = None
    compress: str = "lzw"

    @classmethod
    def from_dict(cls, cfg: dict) -> "TilerConfig":
        """Construct from a plain dictionary (e.g., parsed YAML section).

        Args:
            cfg: Dictionary with keys matching the dataclass field names.
                Unknown keys are silently ignored.

        Returns:
            A populated :class:`TilerConfig` instance.
        """
        return cls(
            tile_size=cfg.get("tile_size", 320),
            overlap=cfg.get("overlap", 64),
            min_percentile=cfg.get("min_percentile", 1.0),
            max_percentile=cfg.get("max_percentile", 99.0),
            bands=cfg.get("bands"),
            compress=cfg.get("compress", "lzw"),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_bands(data: np.ndarray, n_src_bands: int, bands_cfg: Optional[List[int]]) -> np.ndarray:
    """Select and arrange source bands into a 3-channel (R, G, B) array.

    Args:
        data: Raw rasterio read result, shape ``(n_bands, H, W)``.
        n_src_bands: Number of bands in the source dataset.
        bands_cfg: Optional list of 1-based band indices specifying
            which source bands map to (R, G, B). ``None`` triggers
            auto-detection.

    Returns:
        Array of shape ``(3, H, W)`` with dtype preserved from input.

    Raises:
        ValueError: If any requested band index is out of range.
    """
    if bands_cfg is not None:
        for b in bands_cfg:
            if b < 1 or b > n_src_bands:
                raise ValueError(
                    f"Requested band {b} is out of range for a "
                    f"{n_src_bands}-band source image."
                )
        selected = data[[b - 1 for b in bands_cfg]]  # 1-based → 0-based
    elif n_src_bands >= 3:
        selected = data[:3]
    else:
        selected = data[:1]

    # Panchromatic → replicate to RGB so all outputs are consistently 3-channel.
    if selected.shape[0] == 1:
        selected = np.repeat(selected, 3, axis=0)

    return selected  # (3, H, W)


def _percentile_stretch(
    rgb: np.ndarray,
    lo_pct: float,
    hi_pct: float,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Compute global percentile parameters and stretch to uint8.

    The percentiles are computed over the *entire* image before tiling so
    that every derived tile shares the same colour rendering — preventing
    per-tile contrast drift that would otherwise confuse the model.

    Args:
        rgb: Array of shape ``(3, H, W)`` in the source dtype.
        lo_pct: Lower percentile (e.g., 1.0).
        hi_pct: Upper percentile (e.g., 99.0).

    Returns:
        A tuple of:
            - ``uint8_rgb``: Stretched array of shape ``(3, H, W)``
              dtype ``uint8``, with pixel values clipped to [1, 254] to
              avoid confusion with exact-black (nodata) or exact-white.
            - ``stretch_params``: List of ``(lo, hi)`` raw value pairs,
              one per band, for logging / debugging.
    """
    stretch_params: List[Tuple[float, float]] = []
    out = np.empty_like(rgb, dtype=np.float32)

    for i in range(rgb.shape[0]):
        band = rgb[i].astype(np.float32)
        lo   = float(np.percentile(band, lo_pct))
        hi   = float(np.percentile(band, hi_pct))
        stretch_params.append((lo, hi))

        if hi - lo < 1e-6:
            out[i] = 0.0
        else:
            s = (band - lo) / (hi - lo) * 255.0
            out[i] = np.clip(s, 1.0, 254.0)

    return out.astype(np.uint8), stretch_params


def _pad_tile(arr: np.ndarray, tile_size: int) -> np.ndarray:
    """Zero-pad a ``(3, H, W)`` array to ``(3, tile_size, tile_size)``.

    Args:
        arr: Input array with shape ``(3, h, w)`` where ``h ≤ tile_size``
            and ``w ≤ tile_size``.
        tile_size: Target spatial dimension (height and width).

    Returns:
        Zero-padded array of shape ``(3, tile_size, tile_size)``.
    """
    pad_h = tile_size - arr.shape[1]
    pad_w = tile_size - arr.shape[2]
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


def _build_output_profile(
    src: rasterio.DatasetReader,
    tile_transform: Affine,
    tile_size: int,
    compress: str,
) -> dict:
    """Build a rasterio write profile for a uint8 RGB GeoTIFF tile.

    Args:
        src: Open source rasterio dataset (used for CRS).
        tile_transform: Affine transform anchored at the tile's top-left
            corner in the source CRS.
        tile_size: Tile width and height in pixels.
        compress: GeoTIFF compression codec.

    Returns:
        A profile dictionary ready to pass to :func:`rasterio.open`.
    """
    return {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "count":     3,          # always 3-channel (RGB)
        "width":     tile_size,
        "height":    tile_size,
        "crs":       src.crs,
        "transform": tile_transform,
        "compress":  compress,
        "predictor": 2,          # horizontal differencing — improves LZW ratio
        "tiled":     True,
        "blockxsize": min(256, tile_size),
        "blockysize": min(256, tile_size),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ImageTiler:
    """Slides a window over GeoTIFF images and saves each patch as a tile.

    Each output tile is a fully georeferenced GeoTIFF:
        - CRS is copied verbatim from the source image.
        - The Affine transform is recomputed for the tile's pixel origin.
        - TIFF tags record provenance: source filename, column / row
          offsets, and original image dimensions.

    This makes the tile self-describing: no external metadata file is
    needed to relate a tile back to its source or to reconstruct
    geospatial coordinates from pixel positions.

    Args:
        config: :class:`TilerConfig` with all tiling hyperparameters.
    """

    def __init__(self, config: TilerConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def tile_directory(self, raw_dir: Path, output_dir: Path) -> List[Path]:
        """Tile every ``.tif`` file found in *raw_dir*.

        Args:
            raw_dir: Directory containing source GeoTIFF files.
            output_dir: Directory where tile GeoTIFFs will be written.
                Created automatically if it does not exist.

        Returns:
            Flat list of :class:`~pathlib.Path` objects for every tile
            written across all source images.

        Raises:
            FileNotFoundError: If *raw_dir* does not exist.
            RuntimeError: If no ``.tif`` files are found in *raw_dir*.
        """
        if not raw_dir.exists():
            raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")

        tif_files = sorted(raw_dir.glob("*.tif"))
        if not tif_files:
            raise RuntimeError(f"No .tif files found in '{raw_dir}'.")

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Tiling %d source image(s) from '%s' → '%s'",
            len(tif_files), raw_dir, output_dir,
        )

        all_tile_paths: List[Path] = []
        for tif_path in tif_files:
            paths = self.tile_image(tif_path, output_dir)
            all_tile_paths.extend(paths)
            logger.info(
                "  %s → %d tile(s) written.", tif_path.name, len(paths)
            )

        logger.info(
            "Tiling complete. Total tiles written: %d", len(all_tile_paths)
        )
        return all_tile_paths

    def tile_image(self, tif_path: Path, output_dir: Path) -> List[Path]:
        """Tile a single GeoTIFF and return the paths of all written tiles.

        Processing steps:
            1. Read the full image into memory and select RGB bands.
            2. Compute global percentile-stretch parameters (once per image
               so every tile has a consistent colour rendering).
            3. Slide a ``tile_size × tile_size`` window with stride
               ``tile_size - overlap`` over the stretched image.
            4. Skip uniform tiles (all pixels identical → nodata / ocean pad).
            5. Zero-pad edge tiles to ``tile_size × tile_size``.
            6. Write each patch as a georeferenced GeoTIFF with provenance tags.

        Args:
            tif_path: Path to the source GeoTIFF.
            output_dir: Directory to write tile files into.

        Returns:
            List of :class:`~pathlib.Path` objects, one per tile written.

        Raises:
            rasterio.errors.RasterioIOError: If the source file cannot be
                opened by rasterio.
        """
        cfg    = self._cfg
        stride = cfg.tile_size - cfg.overlap

        tile_paths: List[Path] = []

        with rasterio.open(tif_path) as src:
            W, H      = src.width, src.height
            n_bands   = src.count
            stem      = tif_path.stem

            logger.info(
                "  Source: %s  (%d × %d px, %d band(s), CRS: %s)",
                tif_path.name, W, H, n_bands, src.crs,
            )

            # ----------------------------------------------------------
            # 1. Read full image; select RGB bands.
            # ----------------------------------------------------------
            logger.debug("  Reading full image into memory …")
            full_data = src.read()                                  # (B, H, W)
            rgb_full  = _select_bands(full_data, n_bands, cfg.bands)  # (3, H, W)
            del full_data

            # ----------------------------------------------------------
            # 2. Global percentile stretch → uint8 (applied to full img).
            # ----------------------------------------------------------
            logger.debug("  Computing global percentile stretch …")
            rgb_uint8, stretch_params = _percentile_stretch(
                rgb_full, cfg.min_percentile, cfg.max_percentile
            )
            del rgb_full

            for i, (lo, hi) in enumerate(stretch_params):
                logger.debug(
                    "    Band %d: raw range [%.1f, %.1f]", i + 1, lo, hi
                )

            # ----------------------------------------------------------
            # 3. Sliding window.
            # ----------------------------------------------------------
            n_cols  = math.ceil(W / stride)
            n_rows  = math.ceil(H / stride)
            kept    = 0
            skipped = 0

            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    x_off = col_idx * stride
                    y_off = row_idx * stride

                    win_w = min(cfg.tile_size, W - x_off)
                    win_h = min(cfg.tile_size, H - y_off)

                    # Slice from the already-stretched full image.
                    tile_rgb = rgb_uint8[
                        :, y_off : y_off + win_h, x_off : x_off + win_w
                    ].copy()

                    # --------------------------------------------------
                    # 4. Skip uniform (empty / nodata) tiles.
                    # --------------------------------------------------
                    if tile_rgb.min() == tile_rgb.max():
                        skipped += 1
                        continue

                    # --------------------------------------------------
                    # 5. Zero-pad edge tiles.
                    # --------------------------------------------------
                    tile_rgb = _pad_tile(tile_rgb, cfg.tile_size)

                    # --------------------------------------------------
                    # 6. Compute tile-specific affine transform.
                    #    rasterio.window_transform correctly anchors the
                    #    top-left corner of the window in the source CRS.
                    # --------------------------------------------------
                    window         = Window(x_off, y_off, win_w, win_h)
                    tile_transform = src.window_transform(window)

                    profile = _build_output_profile(
                        src, tile_transform, cfg.tile_size, cfg.compress
                    )

                    tile_name = f"{stem}_{x_off}_{y_off}.tif"
                    out_path  = output_dir / tile_name

                    with rasterio.open(out_path, "w", **profile) as dst:
                        dst.write(tile_rgb)
                        dst.update_tags(
                            source_tif=tif_path.name,
                            col_off=str(x_off),
                            row_off=str(y_off),
                            src_width=str(W),
                            src_height=str(H),
                            tile_size=str(cfg.tile_size),
                        )

                    tile_paths.append(out_path)
                    kept += 1

            del rgb_uint8

        logger.info(
            "  → %d tile(s) saved, %d uniform tile(s) skipped.",
            kept, skipped,
        )
        return tile_paths
