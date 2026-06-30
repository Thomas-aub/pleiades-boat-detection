"""
src/vessels_detect/preprocessing/image_enhancement.py
-------------------------------------------------------
Step 1 - Image enhancement: radiometric normalisation + upsampling.

Two sub-operations are applied to every raw GeoTIFF, in order:

1.  **Radiometric normalisation** - percentile clip -> gamma correction ->
    uint8 rescale.  Statistics are computed once from a small thumbnail so
    every pixel of the output shares the same colour rendering.
2.  **Spatial upsampling** - rasterio's ``WarpedVRT`` resamples the image to
    a higher resolution while keeping the geospatial transform correct.

Both operations stream the image block-by-block (rasterio windows), so RAM
use stays flat regardless of source image size.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

logger = logging.getLogger(__name__)

# uint8 output range. Avoids exact 0 (often nodata) and 255 (clipped highlight).
_OUT_LO, _OUT_HI = 1.0, 254.0
_THUMB_LONG_EDGE = 1024

_RESAMPLING_MAP = {
    "lanczos": Resampling.lanczos,
    "cubic": Resampling.cubic,
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
}


# ---------------------------------------------------------------------------
# Radiometric normalisation
# ---------------------------------------------------------------------------

def _select_band_indices(n_src_bands: int, bands_cfg: Optional[List[int]]) -> List[int]:
    """Pick 0-based (R, G, B) source band indices."""
    if bands_cfg is not None:
        return [b - 1 for b in bands_cfg]
    if n_src_bands >= 3:
        return [0, 1, 2]
    return [0, 0, 0]  # panchromatic -> replicate to all 3 channels


def _compute_stretch_params(
    src: rasterio.DatasetReader,
    band_indices: List[int],
    lo_pct: float,
    hi_pct: float,
) -> List[Tuple[float, float]]:
    """Compute per-band (lo, hi) DN anchors from a fast thumbnail read."""
    scale = _THUMB_LONG_EDGE / max(src.width, src.height)
    out_h, out_w = max(int(src.height * scale), 1), max(int(src.width * scale), 1)

    unique_src_bands = sorted(set(b + 1 for b in band_indices))
    thumb = src.read(
        indexes=unique_src_bands,
        out_shape=(len(unique_src_bands), out_h, out_w),
        resampling=Resampling.bilinear,
    )
    pos = {b: i for i, b in enumerate(unique_src_bands)}

    params: List[Tuple[float, float]] = []
    for b_idx in band_indices:
        row = thumb[pos[b_idx + 1]].astype(np.float32)
        valid = row[row > 0]
        if valid.size == 0:
            params.append((0.0, 1.0))
            continue
        lo, hi = float(np.percentile(valid, lo_pct)), float(np.percentile(valid, hi_pct))
        params.append((lo, hi) if hi > lo else (lo, lo + 1.0))
    return params


def _stretch_and_gamma(
    src: rasterio.DatasetReader,
    dst: rasterio.DatasetWriter,
    band_indices: List[int],
    stretch: List[Tuple[float, float]],
    gamma: float,
) -> None:
    """Stream radiometric correction window-by-window (low RAM)."""
    for _, window in dst.block_windows(1):
        out_bands = []
        for b_idx, (lo, hi) in zip(band_indices, stretch):
            raw = src.read(b_idx + 1, window=window).astype(np.float32)
            normalised = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
            corrected = np.power(normalised, gamma)
            scaled = corrected * (_OUT_HI - _OUT_LO) + _OUT_LO
            out_bands.append(np.clip(scaled, _OUT_LO, _OUT_HI).astype(np.uint8))
        dst.write(np.stack(out_bands, axis=0), window=window)


def apply_radiometric_correction(
    src_path: Path,
    dst_path: Path,
    lo_percentile: float,
    hi_percentile: float,
    gamma: float,
    bands: Optional[List[int]],
    compress: str,
) -> None:
    """Percentile-stretch + gamma-correct one GeoTIFF, writing a uint8 RGB output.

    Memory-safe: only a small thumbnail and one rasterio block at a time are
    ever held in memory; the source dataset itself is never fully loaded.

    Skipped entirely (the source file is copied through verbatim) when
    ``lo_percentile == 0`` and ``hi_percentile == 100``, since that range
    means "no clipping" - i.e. the stretch would be a no-op anyway.
    """
    if lo_percentile == 0 and hi_percentile == 100:
        logger.info("  [enhance] radiometric stretch skipped (lo=0, hi=100): %s", src_path.name)
        shutil.copy2(src_path, dst_path)
        return

    with rasterio.open(src_path) as src:
        band_indices = _select_band_indices(src.count, bands)
        stretch = _compute_stretch_params(src, band_indices, lo_percentile, hi_percentile)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff", dtype="uint8", count=3, compress=compress,
            predictor=2, tiled=True, blockxsize=256, blockysize=256,
            photometric="RGB",
        )
        original_tags = src.tags()

        tmp_path = dst_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            _stretch_and_gamma(src, dst, band_indices, stretch, gamma)
            dst.update_tags(**original_tags, radiometric_gamma=str(gamma))

    tmp_path.replace(dst_path)


# ---------------------------------------------------------------------------
# Spatial upsampling
# ---------------------------------------------------------------------------

def _update_affine(old_tf: Affine, scale_x: float, scale_y: float) -> Affine:
    return old_tf * Affine.scale(1.0 / scale_x, 1.0 / scale_y)


def apply_upsampling(
    src_path: Path,
    dst_path: Path,
    upscale_ratio: float,
    interpolation: str,
    compress: str,
) -> None:
    """Resample one GeoTIFF to a higher resolution using a streaming WarpedVRT.

    Memory-safe: the VRT is read and written one rasterio block at a time;
    the full-resolution array is never materialised.
    """
    resampling = _RESAMPLING_MAP[interpolation]

    with rasterio.open(src_path) as src:
        out_w = max(1, round(src.width * upscale_ratio))
        out_h = max(1, round(src.height * upscale_ratio))
        scale_x, scale_y = out_w / src.width, out_h / src.height
        new_tf = _update_affine(src.transform, scale_x, scale_y)

        # No src_crs/dst_crs here: we're only resampling resolution, not
        # reprojecting, and passing identical CRS args makes GDAL emit a
        # spurious "warp options does not support option DST_CRS" warning.
        with WarpedVRT(
            src, width=out_w, height=out_h, transform=new_tf,
            resampling=resampling,
        ) as vrt:
            profile = src.profile.copy()
            profile.update(
                width=out_w, height=out_h, transform=new_tf, compress=compress,
                predictor=2, tiled=True, blockxsize=256, blockysize=256,
                BIGTIFF="YES",
            )
            original_tags = src.tags()

            tmp_path = dst_path.with_suffix(".tmp.tif")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for _, window in dst.block_windows(1):
                    dst.write(vrt.read(window=window), window=window)
                dst.update_tags(**original_tags, resampled="true", new_size=f"{out_w}x{out_h}")

    tmp_path.replace(dst_path)


# ---------------------------------------------------------------------------
# Public entry point - used by scripts/preprocessing.py
# ---------------------------------------------------------------------------

def enhance_image(image_path: Path, out_dir: Path, cfg: Dict) -> Path:
    """Run radiometric normalisation, then upsampling, on one raw GeoTIFF.

    Args:
        image_path: Source raw GeoTIFF.
        out_dir: Directory where the enhanced GeoTIFF is written.
        cfg: Full resolved pipeline config (uses ``cfg["radiometric"]`` and
            ``cfg["spatial"]``).

    Returns:
        Path to the enhanced GeoTIFF.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    radiometric_cfg = cfg["radiometric"]
    spatial_cfg = cfg["spatial"]

    stretched_path = out_dir / f"{image_path.stem}.stretched.tif"
    final_path = out_dir / image_path.name

    logger.info("  [enhance] radiometric: %s", image_path.name)
    apply_radiometric_correction(
        src_path=image_path,
        dst_path=stretched_path,
        lo_percentile=radiometric_cfg.get("lo_percentile", 1.0),
        hi_percentile=radiometric_cfg.get("hi_percentile", 99.9),
        gamma=radiometric_cfg.get("gamma", 0.8),
        bands=radiometric_cfg.get("bands"),
        compress=radiometric_cfg.get("compress", "lzw"),
    )

    logger.info("  [enhance] upsampling: %s", image_path.name)
    apply_upsampling(
        src_path=stretched_path,
        dst_path=final_path,
        upscale_ratio=spatial_cfg.get("upscale_ratio", 1.0),
        interpolation=spatial_cfg.get("interpolation", "cubic"),
        compress=spatial_cfg.get("compress", "lzw"),
    )

    stretched_path.unlink(missing_ok=True)
    return final_path