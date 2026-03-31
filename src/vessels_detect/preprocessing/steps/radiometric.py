"""
src/vessels_detect/preprocessing/steps/radiometric.py
------------------------------------------------------
Stage 1 — Local Contrast Enhancement (CLAHE).

Applies **per-band CLAHE** (Contrast Limited Adaptive Histogram
Equalization) to each raw GeoTIFF, then writes a uint8 RGB GeoTIFF that
preserves all spatial metadata (CRS, Affine transform, TIFF tags).

Memory strategy
~~~~~~~~~~~~~~~
Pleiades Neo scenes (~25 000 × 25 000 px) exceed available RAM when loaded
as float64 for CLAHE.  This step uses a **three-pass approach**:

Pass 1 — thumbnail statistics (cheap)
    A fast bilinear thumbnail is read to compute the global percentile
    anchors ``(lo, hi)`` used for pre-clipping.
    Peak RAM: thumbnail only (~2 048 px long edge).

Pass 2 — streaming pre-clip → mmap temp file
    The full band is read once as float32, pre-clipped and normalised to
    ``[0, 1]``, written to a ``numpy`` memory-mapped file, then the
    in-memory copy is immediately freed.
    Peak RAM during this pass: one band × float32.

Pass 3 — CLAHE on the mmap
    ``skimage.equalize_adapthist`` is called on the mmap array.  The OS
    pages data in/out on demand rather than pinning everything in RAM.
    Peak *resident* RAM: roughly ``kernel_rows × image_width × float64``
    — with ``clahe_kernel_size: [256, 256]`` this stays well under 1 GB
    even for 25k-pixel images.

RAM is logged at every major allocation point (INFO level) so that OOM
issues can be diagnosed from the standard pipeline log.

Output contract (identical to the original percentile-stretch step)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Output dtype:      ``uint8``
* Output band count: ``3`` (RGB)
* Output CRS / Affine: copied verbatim from source
* Output directory:  ``cfg["paths"]["radiometric_dir"]``
* File naming:       identical to source (``<stem>.tif``)

YAML section (``configs/preprocessing.yaml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: yaml

    radiometric:
      lo_percentile:    1.0
      hi_percentile:    99.9
      clahe_clip_limit: 0.03
      clahe_kernel_size: [256, 256]   # explicit is safer than null for large images
      gamma:            1.0
      bands:            null
      compress:         "lzw"
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling

from src.vessels_detect.preprocessing.steps.base import BaseStep

logger = logging.getLogger(__name__)

_OUT_LO: float = 1.0
_OUT_HI: float = 254.0

_THUMB_LONG_EDGE: int = 2048


# ---------------------------------------------------------------------------
# RAM reporting
# ---------------------------------------------------------------------------

def _log_ram(label: str) -> None:
    """Log process RSS and system available RAM at INFO level.

    Reads ``/proc/self/status`` and ``/proc/meminfo`` (Linux).  Both are
    logged at INFO so they appear in normal runs without ``--log-level DEBUG``.

    Args:
        label: Free-text description of the current pipeline point.
    """
    rss_mb: Optional[float]   = None
    avail_mb: Optional[float] = None

    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024
                    break
    except OSError:
        pass

    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) / 1024
                    break
    except OSError:
        pass

    parts = [f"[RAM] {label}"]
    if rss_mb   is not None: parts.append(f"process={rss_mb:.0f} MB")
    if avail_mb is not None: parts.append(f"sys_avail={avail_mb:.0f} MB")
    logger.info("  ".join(parts))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RadiometricConfig:
    """Hyperparameters for :class:`RadiometricStep`.

    Attributes:
        lo_percentile:     Lower clipping percentile before CLAHE.
        hi_percentile:     Upper clipping percentile before CLAHE.
        clahe_clip_limit:  CLAHE contrast-clip limit in ``[0, 1]``.
        clahe_kernel_size: CLAHE tile shape ``(rows, cols)``.
                           **Always set this explicitly for large images.**
                           ``None`` → skimage default = 1/8 of image dims,
                           which is ~3 000 px for a 25k-pixel scene and can
                           trigger OOM.  ``[256, 256]`` is a safe default.
        gamma:             Post-CLAHE gamma; ``1.0`` = identity.
        bands:             Optional 1-based source band indices for RGB.
        compress:          GeoTIFF compression codec.
    """

    lo_percentile:     float                     = 1.0
    hi_percentile:     float                     = 99.9
    clahe_clip_limit:  float                     = 0.03
    clahe_kernel_size: Optional[Tuple[int, int]] = (256, 256)
    gamma:             float                     = 1.0
    bands:             Optional[List[int]]       = None
    compress:          str                       = "lzw"

    @classmethod
    def from_dict(cls, cfg: dict) -> "RadiometricConfig":
        ks_raw = cfg.get("clahe_kernel_size")
        if ks_raw is None:
            kernel_size: Optional[Tuple[int, int]] = None   # explicit opt-in to skimage auto
        elif isinstance(ks_raw, (list, tuple)):
            kernel_size = (int(ks_raw[0]), int(ks_raw[1]))
        else:
            kernel_size = (int(ks_raw), int(ks_raw))

        return cls(
            lo_percentile    = cfg.get("lo_percentile",    1.0),
            hi_percentile    = cfg.get("hi_percentile",    99.9),
            clahe_clip_limit = cfg.get("clahe_clip_limit", 0.03),
            clahe_kernel_size= kernel_size,
            gamma            = cfg.get("gamma",            1.0),
            bands            = cfg.get("bands"),
            compress         = cfg.get("compress",         "lzw"),
        )


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------

def _select_band_indices(n_src_bands: int, bands_cfg: Optional[List[int]]) -> List[int]:
    if bands_cfg is not None:
        for b in bands_cfg:
            if b < 1 or b > n_src_bands:
                raise ValueError(
                    f"Band index {b} out of range for a {n_src_bands}-band source."
                )
        return [b - 1 for b in bands_cfg]
    return [0, 1, 2] if n_src_bands >= 3 else [0, 0, 0]


# ---------------------------------------------------------------------------
# Pass 1 — thumbnail-based percentile statistics
# ---------------------------------------------------------------------------

def _compute_stretch_params(
    src: rasterio.DatasetReader,
    band_indices: List[int],
    lo_pct: float,
    hi_pct: float,
) -> List[Tuple[float, float]]:
    """Compute per-band ``(lo, hi)`` clip anchors from a downsampled thumbnail.

    Args:
        src:          Open rasterio dataset.
        band_indices: 0-based band indices for the three output channels.
        lo_pct:       Lower clipping percentile.
        hi_pct:       Upper clipping percentile.

    Returns:
        List of ``(lo, hi)`` pairs in normalised ``[0, 1]`` space.
    """
    scale = _THUMB_LONG_EDGE / max(src.width, src.height)
    out_h = max(int(src.height * scale), 1)
    out_w = max(int(src.width  * scale), 1)

    unique_src = sorted({b + 1 for b in band_indices})
    thumb = src.read(
        indexes    = unique_src,
        out_shape  = (len(unique_src), out_h, out_w),
        resampling = Resampling.bilinear,
    ).astype(np.float32)

    dtype_info = np.iinfo(src.dtypes[0]) if np.issubdtype(
        np.dtype(src.dtypes[0]), np.integer
    ) else None
    if dtype_info is not None:
        thumb /= float(dtype_info.max)
    else:
        mx = thumb.max()
        if mx > 0:
            thumb /= mx

    pos = {b: i for i, b in enumerate(unique_src)}
    params: List[Tuple[float, float]] = []
    for b_idx in band_indices:
        row   = thumb[pos[b_idx + 1]]
        valid = row[row > 0]
        if valid.size == 0:
            params.append((0.0, 1.0))
            continue
        lo = float(np.percentile(valid, lo_pct))
        hi = float(np.percentile(valid, hi_pct))
        if hi <= lo:
            hi = lo + 1e-6
        params.append((lo, hi))

    logger.info("    Stretch params (lo, hi) per channel: %s", params)
    return params


# ---------------------------------------------------------------------------
# Pass 2 — read full band, pre-clip, persist to mmap temp file
# ---------------------------------------------------------------------------

def _preprocess_band_to_mmap(
    src: rasterio.DatasetReader,
    band_idx: int,
    lo: float,
    hi: float,
    tmp_path: str,
) -> np.ndarray:
    """Read one raw band, pre-clip to ``[lo, hi]``, write to a float32 mmap.

    The in-memory copy is freed immediately after the mmap is written so
    that only the mmap file (OS-managed) occupies space during CLAHE.

    Args:
        src:      Open rasterio dataset.
        band_idx: 0-based band index.
        lo:       Lower clip anchor (normalised ``[0, 1]``).
        hi:       Upper clip anchor (normalised ``[0, 1]``).
        tmp_path: Destination path for the ``.npy`` mmap file.

    Returns:
        Read-write numpy mmap, shape ``(height, width)``, dtype float32.
    """
    h, w = src.height, src.width
    mb   = h * w * 4 / 1024 / 1024
    logger.info("    Band %d: reading raw  (%d × %d px, ~%.0f MB float32)", band_idx, h, w, mb)
    _log_ram(f"before read band {band_idx}")

    dtype_info = np.iinfo(src.dtypes[band_idx]) if np.issubdtype(
        np.dtype(src.dtypes[band_idx]), np.integer
    ) else None

    raw = src.read(band_idx + 1).astype(np.float32)
    _log_ram(f"after read band {band_idx}")

    if dtype_info is not None:
        raw /= float(dtype_info.max)
    else:
        mx = float(raw.max())
        if mx > 0:
            raw /= mx

    raw = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    _log_ram(f"after pre-clip band {band_idx}")

    mmap = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(h, w))
    mmap[:] = raw
    del raw
    gc.collect()
    _log_ram(f"after mmap write band {band_idx} (in-memory array freed)")

    return mmap


# ---------------------------------------------------------------------------
# Pass 3 — CLAHE on mmap → uint8
# ---------------------------------------------------------------------------

def _apply_clahe_on_mmap(
    mmap: np.ndarray,
    clip_limit: float,
    kernel_size: Optional[Tuple[int, int]],
    gamma: float,
    band_idx: int,
) -> np.ndarray:
    """Run CLAHE on a memory-mapped float32 band, return a uint8 result.

    Args:
        mmap:        Float32 mmap, values in ``[0, 1]``.
        clip_limit:  CLAHE contrast-clip limit.
        kernel_size: CLAHE tile shape or ``None`` for skimage default.
        gamma:       Post-CLAHE gamma exponent.
        band_idx:    Band index (for logging only).

    Returns:
        uint8 array in ``[1, 254]``, shape ``(H, W)``.
    """
    from skimage.exposure import equalize_adapthist

    ks_log = str(kernel_size) if kernel_size else "auto (1/8 of dims — may OOM on large images)"
    logger.info(
        "    Band %d: CLAHE  clip_limit=%.3f  kernel_size=%s",
        band_idx, clip_limit, ks_log,
    )
    _log_ram(f"before CLAHE band {band_idx}")

    # skimage requires float64 in [0,1] — one unavoidable copy.
    enhanced = equalize_adapthist(
        mmap.astype(np.float64),
        kernel_size = kernel_size,
        clip_limit  = clip_limit,
        nbins       = 256,
    )
    _log_ram(f"after CLAHE band {band_idx}")

    if abs(gamma - 1.0) > 1e-6:
        enhanced = np.power(enhanced, gamma)

    result = np.clip(enhanced * (_OUT_HI - _OUT_LO) + _OUT_LO, _OUT_LO, _OUT_HI).astype(np.uint8)
    del enhanced
    gc.collect()
    _log_ram(f"after uint8 cast band {band_idx}")

    return result


# ---------------------------------------------------------------------------
# Output profile
# ---------------------------------------------------------------------------

def _build_profile(src: rasterio.DatasetReader, compress: str) -> dict:
    profile = src.profile.copy()
    profile.update(
        driver      = "GTiff",
        dtype       = "uint8",
        count       = 3,
        compress    = compress,
        predictor   = 2,
        tiled       = True,
        blockxsize  = 256,
        blockysize  = 256,
        photometric = "RGB",
    )
    return profile


# ---------------------------------------------------------------------------
# Public step
# ---------------------------------------------------------------------------

class RadiometricStep(BaseStep):
    """Apply CLAHE local contrast enhancement to every raw GeoTIFF."""

    NAME = "radiometric"

    def run(self, cfg: dict) -> None:
        params  = RadiometricConfig.from_dict(cfg.get("radiometric", {}))
        raw_dir = cfg["paths"]["raw_dir"]
        out_dir = cfg["paths"]["radiometric_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(raw_dir.glob("*.tif"))
        if not tif_files:
            raise RuntimeError(f"No .tif files found in '{raw_dir}'.")

        logger.info(
            "CLAHE contrast enhancement: %d image(s)  "
            "clip_limit=%.3f  kernel_size=%s  γ=%.2f",
            len(tif_files),
            params.clahe_clip_limit,
            params.clahe_kernel_size or "auto",
            params.gamma,
        )
        _log_ram("stage start")

        for tif_path in tif_files:
            out_path = out_dir / tif_path.name
            logger.info("  Processing: %s", tif_path.name)
            self._process_image(tif_path, out_path, params)

        logger.info("Radiometric (CLAHE) stage complete → '%s'.", out_dir)

    def _process_image(
        self,
        src_path: Path,
        dst_path: Path,
        params: RadiometricConfig,
    ) -> None:
        tmp_files: List[str] = []

        with rasterio.open(src_path) as src:
            band_mb = src.width * src.height * 4 / 1024 / 1024
            logger.info(
                "  Image: %d × %d px  bands=%d  dtype=%s",
                src.width, src.height, src.count, src.dtypes[0],
            )
            logger.info(
                "  Per-band float32: %.0f MB  |  3-band stack: %.0f MB  |  "
                "CLAHE peak (float64): ~%.0f MB",
                band_mb, band_mb * 3, band_mb * 2,   # float64 = 2× float32
            )

            band_indices  = _select_band_indices(src.count, params.bands)
            stretch       = _compute_stretch_params(
                src, band_indices,
                params.lo_percentile, params.hi_percentile,
            )
            profile       = _build_profile(src, params.compress)
            original_tags = src.tags()

            out_bands: List[np.ndarray] = []

            for ch, (b_idx, (lo, hi)) in enumerate(zip(band_indices, stretch)):
                # Pass 2.
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=f"_clahe_band{ch}.npy")
                os.close(tmp_fd)
                tmp_files.append(tmp_path)

                mmap = _preprocess_band_to_mmap(src, b_idx, lo, hi, tmp_path)

                # Pass 3.
                uint8_band = _apply_clahe_on_mmap(
                    mmap,
                    clip_limit  = params.clahe_clip_limit,
                    kernel_size = params.clahe_kernel_size,
                    gamma       = params.gamma,
                    band_idx    = ch,
                )
                out_bands.append(uint8_band)

                del mmap
                gc.collect()
                _log_ram(f"after releasing mmap band {ch}")

        # Write output.
        stacked = np.stack(out_bands, axis=0)
        logger.info(
            "  Writing output: %s  %.0f MB",
            stacked.shape, stacked.nbytes / 1024 / 1024,
        )
        _log_ram("before writing GeoTIFF")

        tmp_out = dst_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp_out, "w", **profile) as dst:
            dst.write(stacked)
            dst.update_tags(
                **original_tags,
                radiometric_method = "clahe",
                clahe_clip_limit   = str(params.clahe_clip_limit),
                clahe_kernel_size  = str(params.clahe_kernel_size or "auto"),
                radiometric_gamma  = str(params.gamma),
            )
        tmp_out.replace(dst_path)

        del out_bands, stacked
        gc.collect()
        _log_ram("after writing GeoTIFF (arrays freed)")

        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

        logger.info("    → %s", dst_path.name)