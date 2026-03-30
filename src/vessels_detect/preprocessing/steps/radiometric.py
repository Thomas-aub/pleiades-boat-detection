"""
src/vessels_detect/preprocessing/steps/radiometric.py
------------------------------------------------------
Stage 1 — Radiometric Normalisation.

Applies a **global** percentile stretch followed by gamma correction to each
raw GeoTIFF, then writes a uint8 RGB GeoTIFF that preserves all spatial
metadata (CRS, Affine transform, TIFF tags).

Design goals
~~~~~~~~~~~~
* **Global statistics** — percentiles are computed from a fast thumbnail
  read so that every pixel of the output shares a single, consistent colour
  rendering.  Per-tile contrast drift (which confuses feature extractors) is
  eliminated.
* **Streaming I/O** — full-resolution data is written band-by-band via
  rasterio windowed I/O to avoid loading the entire image into RAM.
* **Metadata preservation** — CRS, Affine transform, and all existing TIFF
  tags are copied verbatim to the output; a ``radiometric_params`` tag is
  appended for provenance.

Coordinate flow
~~~~~~~~~~~~~~~
::

    Raw DN  →  clip [lo, hi]  →  stretch to [0, 1]  →  gamma(x) = x^γ
             →  scale to [1, 254]  →  uint8 output

Typical usage (via manager)::

    from src.vessels_detect.preprocessing.steps.radiometric import RadiometricStep
    step = RadiometricStep()
    step.run(cfg)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from src.vessels_detect.preprocessing.steps.base import BaseStep

logger = logging.getLogger(__name__)

# Pixel value range for uint8 output.
# We avoid exact 0 (often used as nodata) and 255 (often clipped highlight).
_OUT_LO: float = 1.0
_OUT_HI: float = 254.0

# Resolution of the thumbnail used for global statistics (longest edge).
_THUMB_LONG_EDGE: int = 1024


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class RadiometricConfig:
    """Hyperparameters for :class:`RadiometricStep`.

    Attributes:
        lo_percentile: Lower clipping percentile (e.g. ``1.0``).
        hi_percentile: Upper clipping percentile (e.g. ``99.9``).
        gamma: Gamma exponent applied after normalisation.  Values below
            1.0 brighten shadows (recommended for vessel detection over
            water).
        bands: Optional list of 1-based source band indices to use as
            ``(R, G, B)``.  ``None`` triggers auto-detection.
        compress: GeoTIFF compression codec for output files.
        window_size: Internal tile size for windowed rasterio writes.
    """

    lo_percentile: float = 1.0
    hi_percentile: float = 99.9
    gamma: float = 0.8
    bands: Optional[List[int]] = None
    compress: str = "lzw"
    window_size: int = 512

    @classmethod
    def from_dict(cls, cfg: dict) -> "RadiometricConfig":
        """Construct from a YAML section dictionary.

        Args:
            cfg: Dictionary with keys matching the dataclass field names.

        Returns:
            A populated :class:`RadiometricConfig` instance.
        """
        return cls(
            lo_percentile=cfg.get("lo_percentile", 1.0),
            hi_percentile=cfg.get("hi_percentile", 99.9),
            gamma=cfg.get("gamma", 0.8),
            bands=cfg.get("bands"),
            compress=cfg.get("compress", "lzw"),
            window_size=cfg.get("window_size", 512),
        )


# ---------------------------------------------------------------------------
# Band selection helper
# ---------------------------------------------------------------------------

def _select_band_indices(n_src_bands: int, bands_cfg: Optional[List[int]]) -> List[int]:
    """Return 0-based band indices for the three output channels.

    Args:
        n_src_bands: Number of bands in the source dataset.
        bands_cfg: Optional 1-based band indices from config.

    Returns:
        List of three 0-based indices.

    Raises:
        ValueError: If a requested band index is out of range.
    """
    if bands_cfg is not None:
        for b in bands_cfg:
            if b < 1 or b > n_src_bands:
                raise ValueError(
                    f"Band index {b} is out of range for a {n_src_bands}-band source."
                )
        return [b - 1 for b in bands_cfg]

    if n_src_bands >= 3:
        return [0, 1, 2]

    # Panchromatic — replicate to all three channels.
    return [0, 0, 0]


# ---------------------------------------------------------------------------
# Global statistics via thumbnail
# ---------------------------------------------------------------------------

def _compute_stretch_params(
    src: rasterio.DatasetReader,
    band_indices: List[int],
    lo_pct: float,
    hi_pct: float,
) -> List[Tuple[float, float]]:
    """Compute per-band (lo, hi) stretch anchors from a thumbnail.

    A fast bilinear thumbnail is read once; valid (non-zero) pixels are
    used to compute the requested percentiles.  This ensures that all
    windows processed later share an identical colour rendering.

    Args:
        src: Open rasterio dataset.
        band_indices: 0-based band indices for the three output channels.
        lo_pct: Lower clipping percentile.
        hi_pct: Upper clipping percentile.

    Returns:
        List of ``(lo, hi)`` raw-DN pairs, one per output channel.
    """
    scale = _THUMB_LONG_EDGE / max(src.width, src.height)
    out_h = max(int(src.height * scale), 1)
    out_w = max(int(src.width  * scale), 1)

    # Read all required source bands in one call (may contain duplicates for
    # panchromatic case — that is fine, percentiles will be identical).
    unique_src_bands = sorted(set(b + 1 for b in band_indices))  # 1-based
    thumb = src.read(
        indexes=unique_src_bands,
        out_shape=(len(unique_src_bands), out_h, out_w),
        resampling=Resampling.bilinear,
    )

    # Map 1-based source index → position in ``thumb`` array.
    src_band_pos: Dict[int, int] = {b: i for i, b in enumerate(unique_src_bands)}

    params: List[Tuple[float, float]] = []
    for b_idx in band_indices:
        row = thumb[src_band_pos[b_idx + 1]].astype(np.float32)
        valid = row[row > 0]

        if valid.size == 0:
            params.append((0.0, 1.0))
            continue

        lo = float(np.percentile(valid, lo_pct))
        hi = float(np.percentile(valid, hi_pct))
        if hi <= lo:
            hi = lo + 1.0
        params.append((lo, hi))

    logger.debug("Stretch params (lo, hi) per channel: %s", params)
    return params


# ---------------------------------------------------------------------------
# Output profile builder
# ---------------------------------------------------------------------------

def _build_profile(src: rasterio.DatasetReader, compress: str) -> dict:
    """Build a rasterio write profile for a uint8 3-band GeoTIFF.

    Args:
        src: Open source dataset (CRS and Affine are copied verbatim).
        compress: GeoTIFF compression codec.

    Returns:
        Profile dictionary ready for :func:`rasterio.open`.
    """
    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=3,
        compress=compress,
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        photometric="RGB",
    )
    return profile


# ---------------------------------------------------------------------------
# Public step
# ---------------------------------------------------------------------------

class RadiometricStep(BaseStep):
    """Apply global percentile stretching + gamma to every raw GeoTIFF.

    For each ``.tif`` file in ``cfg["paths"]["raw_dir"]``, this step:

    1. Reads a thumbnail to compute global per-band percentile anchors.
    2. Streams the full image through a normalise → gamma → uint8 transform
       using rasterio windowed I/O.
    3. Writes the result to ``cfg["paths"]["radiometric_dir"]``, preserving
       CRS, Affine transform, and all TIFF tags.
    """

    NAME = "radiometric"

    def run(self, cfg: dict) -> None:
        """Execute the radiometric normalisation stage.

        Args:
            cfg: Resolved configuration dictionary.  Uses:

                * ``cfg["paths"]["raw_dir"]`` — source GeoTIFFs.
                * ``cfg["paths"]["radiometric_dir"]`` — output directory.
                * ``cfg["radiometric"]`` — stage hyperparameters.
        """
        paths  = cfg["paths"]
        params = RadiometricConfig.from_dict(cfg.get("radiometric", {}))

        raw_dir  : Path = paths["raw_dir"]
        out_dir  : Path = paths["radiometric_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(raw_dir.glob("*.tif"))
        if not tif_files:
            raise RuntimeError(f"No .tif files found in '{raw_dir}'.")

        logger.info(
            "Radiometric normalisation: %d image(s)  lo=%.1f%%  hi=%.1f%%  γ=%.2f",
            len(tif_files), params.lo_percentile, params.hi_percentile, params.gamma,
        )

        for tif_path in tif_files:
            out_path = out_dir / tif_path.name
            logger.info("  Processing: %s", tif_path.name)
            self._process_image(tif_path, out_path, params)

        logger.info("Radiometric stage complete → '%s'.", out_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_image(
        self,
        src_path: Path,
        dst_path: Path,
        params: RadiometricConfig,
    ) -> None:
        """Normalise one GeoTIFF and write to *dst_path*.

        Args:
            src_path: Path to the raw GeoTIFF.
            dst_path: Path for the normalised output GeoTIFF.
            params: :class:`RadiometricConfig` hyperparameters.
        """
        with rasterio.open(src_path) as src:
            band_indices = _select_band_indices(src.count, params.bands)
            stretch      = _compute_stretch_params(
                src, band_indices,
                params.lo_percentile, params.hi_percentile,
            )
            profile = _build_profile(src, params.compress)
            original_tags = src.tags()

            tmp_path = dst_path.with_suffix(".tmp.tif")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                self._stream_windows(
                    src, dst, band_indices, stretch, params.gamma, params.window_size
                )
                # Preserve all original TIFF tags and append provenance.
                provenance = ";".join(
                    f"ch{i}:lo={lo:.2f},hi={hi:.2f}"
                    for i, (lo, hi) in enumerate(stretch)
                )
                dst.update_tags(
                    **original_tags,
                    radiometric_stretch=provenance,
                    radiometric_gamma=str(params.gamma),
                )

        tmp_path.replace(dst_path)
        logger.debug("    → %s", dst_path.name)

    @staticmethod
    def _stream_windows(
        src: rasterio.DatasetReader,
        dst: rasterio.DatasetWriter,
        band_indices: List[int],
        stretch: List[Tuple[float, float]],
        gamma: float,
        window_size: int,
    ) -> None:
        """Write normalised data in rasterio internal tiles.

        Iterates over rasterio's block windows (or a custom grid if the
        source is not tiled) to avoid loading the entire image into RAM.

        Args:
            src: Open source dataset.
            dst: Open destination dataset (same spatial extent).
            band_indices: 0-based source band indices for the 3 output channels.
            stretch: Per-channel ``(lo, hi)`` DN anchors.
            gamma: Gamma exponent.
            window_size: Fallback tile size when source has no internal tiles.
        """
        # Use the destination block grid (guaranteed to be tiled).
        for _, window in dst.block_windows(1):
            out_bands: List[np.ndarray] = []

            for ch_idx, (b_idx, (lo, hi)) in enumerate(zip(band_indices, stretch)):
                raw = src.read(b_idx + 1, window=window).astype(np.float32)

                # 1. Clip + normalise to [0, 1].
                normalised = (raw - lo) / (hi - lo)
                normalised = np.clip(normalised, 0.0, 1.0)

                # 2. Gamma correction.
                corrected = np.power(normalised, gamma)

                # 3. Scale to uint8 range [1, 254].
                scaled = corrected * (_OUT_HI - _OUT_LO) + _OUT_LO
                out_bands.append(np.clip(scaled, _OUT_LO, _OUT_HI).astype(np.uint8))

            dst.write(np.stack(out_bands, axis=0), window=window)
