"""
src/vessels_detect/preprocessing/steps/spatial.py
--------------------------------------------------
Stage 2 — Spatial Resampling (Upsampling / Downsampling).

Rescales each GeoTIFF using a user-defined `upscale_ratio` while preserving
geospatial integrity via rasterio's WarpedVRT API.

Key Features
~~~~~~~~~~~~
- True control over scaling (no implicit downsampling)
- High-quality Lanczos / Cubic interpolation
- Block-wise streaming (low RAM)
- Correct affine transform update
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from src.vessels_detect.preprocessing.steps.base import BaseStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resampling lookup
# ---------------------------------------------------------------------------
_RESAMPLING_MAP = {
    "lanczos":  Resampling.lanczos,
    "cubic":    Resampling.cubic,
    "bilinear": Resampling.bilinear,
    "nearest":  Resampling.nearest,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SpatialConfig:
    upscale_ratio: float = 1.0
    interpolation: str = "lanczos"
    window_size: int = 512
    compress: str = "lzw"

    def __post_init__(self) -> None:
        if self.upscale_ratio <= 0:
            raise ValueError("upscale_ratio must be > 0")

        if self.interpolation not in _RESAMPLING_MAP:
            raise ValueError(
                f"Unknown interpolation '{self.interpolation}'. "
                f"Valid options: {sorted(_RESAMPLING_MAP)}."
            )

    @classmethod
    def from_dict(cls, cfg: dict) -> "SpatialConfig":
        return cls(
            upscale_ratio=cfg.get("upscale_ratio", 1.0),
            interpolation=cfg.get("interpolation", "lanczos"),
            window_size=cfg.get("window_size", 512),
            compress=cfg.get("compress", "lzw"),
        )

    @property
    def resampling(self) -> Resampling:
        return _RESAMPLING_MAP[self.interpolation]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _compute_output_shape(
    src_width: int,
    src_height: int,
    scale: float,
) -> Tuple[int, int]:
    out_w = max(1, int(round(src_width * scale)))
    out_h = max(1, int(round(src_height * scale)))
    return out_w, out_h


def _update_affine(old_tf: Affine, scale_x: float, scale_y: float) -> Affine:
    """
    Adjust affine so the geographic extent stays identical
    while pixel size changes.
    """
    return old_tf * Affine.scale(1.0 / scale_x, 1.0 / scale_y)


def _build_profile(
    src: rasterio.DatasetReader,
    out_w: int,
    out_h: int,
    new_tf: Affine,
    compress: str,
) -> dict:
    profile = src.profile.copy()
    profile.update(
        width=out_w,
        height=out_h,
        transform=new_tf,
        compress=compress,
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="YES",
    )
    return profile


# ---------------------------------------------------------------------------
# Main Step
# ---------------------------------------------------------------------------
class SpatialStep(BaseStep):
    """Resample each GeoTIFF using WarpedVRT."""

    NAME = "spatial"

    def run(self, cfg: dict) -> None:
        paths = cfg["paths"]
        params = SpatialConfig.from_dict(cfg.get("spatial", {}))

        src_dir: Path = paths["radiometric_dir"]
        out_dir: Path = paths["spatial_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(src_dir.glob("*.tif"))
        if not tif_files:
            raise RuntimeError(f"No .tif files found in '{src_dir}'.")

        logger.info(
            "Spatial resampling (VRT): %d image(s) | scale=%.3f | interpolation=%s",
            len(tif_files),
            params.upscale_ratio,
            params.interpolation,
        )

        for tif_path in tif_files:
            out_path = out_dir / tif_path.name
            logger.info("  Processing: %s", tif_path.name)
            self._resample_image(tif_path, out_path, params)

        logger.info("Spatial stage complete → '%s'.", out_dir)

    # -----------------------------------------------------------------------
    def _resample_image(
        self,
        src_path: Path,
        dst_path: Path,
        params: SpatialConfig,
    ) -> None:
        with rasterio.open(src_path) as src:
            scale = params.upscale_ratio

            out_w, out_h = _compute_output_shape(
                src.width, src.height, scale
            )

            scale_x = out_w / src.width
            scale_y = out_h / src.height

            new_tf = _update_affine(src.transform, scale_x, scale_y)

            logger.debug(
                "    Input: %dx%d | Output: %dx%d | scale=%.3f",
                src.width,
                src.height,
                out_w,
                out_h,
                scale,
            )

            # -------------------------------------------------------------------
            # WarpedVRT handles global resampling correctly
            # -------------------------------------------------------------------
            with WarpedVRT(
                src,
                width=out_w,
                height=out_h,
                transform=new_tf,
                resampling=params.resampling,
                src_crs=src.crs,
                dst_crs=src.crs,
            ) as vrt:

                profile = _build_profile(
                    src, out_w, out_h, new_tf, params.compress
                )

                original_tags = src.tags()
                tmp_path = dst_path.with_suffix(".tmp.tif")

                with rasterio.open(tmp_path, "w", **profile) as dst:
                    # Stream block-by-block (low memory)
                    for _, window in dst.block_windows(1):
                        block = vrt.read(window=window)
                        dst.write(block, window=window)

                    dst.update_tags(
                        **original_tags,
                        resampled="true",
                        scale_factor=f"{scale:.3f}",
                        interpolation=params.interpolation,
                        new_size=f"{out_w}x{out_h}",
                    )

        tmp_path.replace(dst_path)
        logger.debug("    → %s", dst_path.name)