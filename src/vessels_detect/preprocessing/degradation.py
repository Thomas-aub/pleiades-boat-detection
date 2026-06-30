"""
src/vessels_detect/preprocessing/degradation.py
---------------------------------------------------
Step 0 - GSD degradation: anti-alias blur + downsampling.

Brings raw imagery from its native ground sample distance (GSD) down to the
target sensor's GSD (e.g. Pléiades Neo 0.3 m/px -> Pléiades 0.5 m/px), so
the rest of the pipeline always sees a consistent resolution domain.

Two things happen, in order:

1.  **Anti-alias blur** - a Gaussian blur sized to the downsampling ratio,
    applied *before* decimation to suppress high-frequency content that
    would otherwise alias into the lower-resolution output. The blur is
    nodata-aware (mask-weighted) so black/nodata borders don't bleed colour
    into valid pixels near the edge.
2.  **Decimation** - block-mean downsampling (equivalent to OpenCV's
    ``INTER_AREA``), the correct choice for shrinking an image since it
    averages every source pixel into the output rather than discarding most
    of them like nearest/bilinear would.

GeoJSON labels are georeferenced in real-world CRS coordinates, so they are
copied through unchanged - only the pixel grid changes, not the geometry.

Memory safety
~~~~~~~~~~~~~
The image is streamed in horizontal row-strips rather than loaded whole.
Each strip is padded with a few extra rows of context (the blur kernel's
support) read from the neighbouring strip, so the Gaussian blur stays
correct at strip boundaries without needing the full image in RAM.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# Row-strip height used for streaming I/O. Large enough to amortise I/O
# overhead, small enough to keep RAM use low on big GeoTIFFs.
_STRIP_HEIGHT = 2048


# ---------------------------------------------------------------------------
# Blur + decimation kernels
# ---------------------------------------------------------------------------

def _sigma_for_ratio(gsd_ratio: float, blur_factor: float) -> float:
    """Gaussian sigma scaled to the downsampling ratio.

    ``blur_factor`` controls aggressiveness; ``0.5 * ratio`` is a common
    rule of thumb that keeps roughly one output pixel's worth of blur.
    """
    return max(blur_factor * gsd_ratio, 0.0)


def _blur_masked(strip: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a (C, H, W) strip, weighting out nodata (all-zero) pixels.

    Pixels are blurred as ``sum(valid * value) / sum(valid)`` so that
    nodata borders don't darken nearby valid pixels.
    """
    if sigma <= 0:
        return strip

    valid_mask = (strip.max(axis=0, keepdims=True) != 0).astype(np.float32)
    numerator = gaussian_filter(strip * valid_mask, sigma=[0, sigma, sigma])
    denominator = gaussian_filter(valid_mask, sigma=[0, sigma, sigma])

    with np.errstate(invalid="ignore", divide="ignore"):
        blurred = np.where(denominator > 0, numerator / denominator, 0.0)
    return blurred


def _decimate_block_mean(strip: np.ndarray, row_edges: np.ndarray, col_edges: np.ndarray) -> np.ndarray:
    """Downsample a (C, H, W) strip by block-averaging (anti-alias safe).

    Equivalent to ``cv2.INTER_AREA`` for integer-ish scale factors: every
    output pixel is the mean of its corresponding source block, so no
    source data is silently discarded.

    Args:
        strip: Source data for this strip, shape ``(C, H, W)``.
        row_edges: Source-row boundaries (length ``out_h + 1``) for this
            strip's slice of the image, relative to ``strip``'s own rows.
        col_edges: Source-column boundaries (length ``out_w + 1``),
            identical for every strip since columns aren't split.
    """
    c = strip.shape[0]
    out_h, out_w = len(row_edges) - 1, len(col_edges) - 1

    out = np.empty((c, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        r0, r1 = row_edges[i], max(row_edges[i] + 1, row_edges[i + 1])
        for j in range(out_w):
            c0, c1 = col_edges[j], max(col_edges[j] + 1, col_edges[j + 1])
            out[:, i, j] = strip[:, r0:r1, c0:c1].mean(axis=(1, 2))
    return out


# ---------------------------------------------------------------------------
# Streaming strip processing
# ---------------------------------------------------------------------------

def _compute_output_shape(src_w: int, src_h: int, scale: float) -> Tuple[int, int]:
    return max(1, round(src_w / scale)), max(1, round(src_h / scale))


def _process_strips(
    src: rasterio.DatasetReader,
    dst: rasterio.DatasetWriter,
    out_w: int,
    out_h: int,
    sigma: float,
) -> None:
    """Stream the image in row-strips: read with halo -> blur -> decimate -> write.

    Only one strip (a small fraction of the full image) is ever held in
    memory, regardless of source resolution.

    A single global row-edge mapping (``out_h + 1`` source-row boundaries)
    is computed once up front and sliced per strip. This guarantees the
    total rows written across all strips sum to exactly ``out_h``, with no
    rounding drift from strip to strip.
    """
    halo = math.ceil(3 * sigma) if sigma > 0 else 0

    all_row_edges = np.linspace(0, src.height, out_h + 1).round().astype(int)
    col_edges = np.linspace(0, src.width, out_w + 1).round().astype(int)

    out_dtype = dst.dtypes[0]
    dtype_max = np.iinfo(out_dtype).max if np.issubdtype(np.dtype(out_dtype), np.integer) else None

    out_row = 0
    while out_row < out_h:
        # Find how many output rows fit in one source strip of _STRIP_HEIGHT.
        in_row_start = all_row_edges[out_row]
        out_row_end = out_row
        while out_row_end < out_h and all_row_edges[out_row_end + 1] - in_row_start <= _STRIP_HEIGHT:
            out_row_end += 1
        out_row_end = max(out_row_end, out_row + 1)  # always make progress

        in_row_end = all_row_edges[out_row_end]
        strip_row_edges = all_row_edges[out_row: out_row_end + 1] - in_row_start

        halo_top = min(halo, in_row_start)
        halo_bottom = min(halo, src.height - in_row_end)

        window = Window(0, in_row_start - halo_top, src.width, (in_row_end - in_row_start) + halo_top + halo_bottom)
        raw_strip = src.read(window=window).astype(np.float32)

        blurred = _blur_masked(raw_strip, sigma)

        # Drop the halo rows before decimating.
        core = blurred[:, halo_top: blurred.shape[1] - halo_bottom, :]
        decimated = _decimate_block_mean(core, strip_row_edges, col_edges)

        if dtype_max is not None:
            decimated = np.clip(decimated, 0, dtype_max)

        write_h = decimated.shape[1]
        out_window = Window(0, out_row, out_w, write_h)
        dst.write(decimated.astype(out_dtype), window=out_window)

        out_row = out_row_end


# ---------------------------------------------------------------------------
# Public entry point - used by scripts/preprocessing.py
# ---------------------------------------------------------------------------

def degrade_image(image_path: Path, out_dir: Path, cfg: Dict) -> Path:
    """Anti-alias blur, then GSD-downsample one raw GeoTIFF.

    Args:
        image_path: Source raw GeoTIFF (native GSD).
        out_dir: Directory where the degraded GeoTIFF is written.
        cfg: Full resolved pipeline config (uses ``cfg["degradation"]``).

    Returns:
        Path to the degraded GeoTIFF. If the stage is disabled
        (``cfg["degradation"]["enabled"] = false``), the source file is
        copied through unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    params = cfg.get("degradation", {})
    out_path = out_dir / image_path.name

    if not params.get("enabled", True):
        logger.info("  [degradation] disabled - copying through: %s", image_path.name)
        shutil.copy2(image_path, out_path)
        _copy_geojson(image_path, out_path)
        return out_path

    source_gsd = params.get("source_gsd_m", 0.3)
    target_gsd = params.get("target_gsd_m", 0.5)
    blur_factor = params.get("blur_factor", 0.45)
    compress = params.get("compress", "deflate")

    scale = target_gsd / source_gsd
    sigma = _sigma_for_ratio(scale, blur_factor)

    logger.info(
        "  [degradation] %s: GSD %.2f -> %.2f m (scale=%.3f, sigma=%.2f)",
        image_path.name, source_gsd, target_gsd, scale, sigma,
    )

    with rasterio.open(image_path) as src:
        out_w, out_h = _compute_output_shape(src.width, src.height, scale)
        out_transform = src.transform * Affine.scale(
            src.width / out_w, src.height / out_h
        )

        profile = src.profile.copy()
        profile.update(
            width=out_w, height=out_h, transform=out_transform,
            compress=compress, predictor=2, tiled=True,
            blockxsize=256, blockysize=256,
        )

        tmp_path = out_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            _process_strips(src, dst, out_w, out_h, sigma)
            dst.update_tags(
                **src.tags(),
                source_gsd_m=str(source_gsd),
                output_gsd_m=str(target_gsd),
            )

    tmp_path.replace(out_path)
    _copy_geojson(image_path, out_path)
    return out_path


def _copy_geojson(src_image_path: Path, dst_image_path: Path) -> None:
    """Copy the matching .geojson alongside the degraded output, unchanged.

    Labels are georeferenced in real-world CRS coordinates, so resampling
    the raster's pixel grid doesn't require touching the annotation file.
    """
    for ext in (".geojson", ".json"):
        geojson_src = src_image_path.with_suffix(ext)
        if geojson_src.exists():
            shutil.copy2(geojson_src, dst_image_path.with_suffix(ext))
            return
    logger.debug("  [degradation] No GeoJSON found for %s.", src_image_path.name)