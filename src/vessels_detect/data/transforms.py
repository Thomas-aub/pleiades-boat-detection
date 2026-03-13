"""
src/vessels_detect/data/transforms.py
-----------------------
Upsamples GeoTIFF tile images from *tile_size* to *target_size* using a
high-quality interpolation filter, while **preserving all GeoTIFF spatial
metadata** (CRS, Affine transform, TIFF tags).

Why pre-upsample instead of relying on YOLO's ``imgsz``?
    YOLO applies a fast bilinear resize at every forward pass when
    ``imgsz > stored_size``.  A single offline Lanczos-4 or Bicubic
    upsample produces sharper edges and richer texture, yielding better
    gradient signal during training and consistent quality at inference.
    Pre-upsampling also removes per-batch resize overhead.

Affine Transform Update
    YOLO OBB label coordinates are normalised [0, 1] fractions, so labels
    remain valid after upsampling without modification.  However, the tile's
    Affine transform must be updated to reflect the new pixel resolution::

        new_transform = old_transform * Affine.scale(old_size / new_size)

    This keeps ``new_transform * (col, row) == old_transform * (col × scale, row × scale)``,
    meaning the top-left corner stays pinned in the same geographic location
    while each pixel now covers a smaller ground footprint.

Parallelism
    All available CPU cores are used via :class:`~concurrent.futures.ProcessPoolExecutor`.
    The worker function :func:`_resize_worker` is defined at module level so
    it can be pickled by the executor.

Typical usage::

    from src.data.transforms import ImageUpsampler, UpsampleConfig

    config = UpsampleConfig(tile_size=320, target_size=640)
    upsampler = ImageUpsampler(config)
    upsampler.upsample_splits(
        images_dir=Path("data/processed/images"),
        splits=["train", "val"],
    )
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from affine import Affine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interpolation name → cv2 constant mapping.
# Used both for config validation and for the worker (which receives the
# integer constant directly so it does not need to import this module).
# ---------------------------------------------------------------------------
_INTERP_MAP: Dict[str, int] = {
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
    "INTER_CUBIC":    cv2.INTER_CUBIC,
    "INTER_LINEAR":   cv2.INTER_LINEAR,
    "INTER_NEAREST":  cv2.INTER_NEAREST,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class UpsampleConfig:
    """Hyperparameters for :class:`ImageUpsampler`.

    Attributes:
        tile_size: Source tile size in pixels (must match tiling step).
            Used to compute the Affine scale factor when rewriting each
            tile's spatial metadata.
        target_size: Desired output size in pixels.  Must be strictly
            larger than ``tile_size``.
        interpolation: Name of the cv2 interpolation method.  One of
            ``"INTER_LANCZOS4"`` (sharpest, slight ringing on hard edges),
            ``"INTER_CUBIC"`` (good quality, no ringing), or
            ``"INTER_LINEAR"`` (fast bilinear, noticeably softer).
        overwrite: If ``True``, each tile is replaced in-place.  If
            ``False``, a new file with the suffix ``_<target_size>`` is
            written beside the original.
        max_workers: Number of parallel worker processes.  ``None`` uses
            all logical CPU cores reported by :func:`os.cpu_count`.
    """

    tile_size: int          = 320
    target_size: int        = 640
    interpolation: str      = "INTER_CUBIC"
    overwrite: bool         = True
    max_workers: Optional[int] = None

    def __post_init__(self) -> None:
        if self.target_size <= self.tile_size:
            raise ValueError(
                f"target_size ({self.target_size}) must be greater than "
                f"tile_size ({self.tile_size})."
            )
        if self.interpolation not in _INTERP_MAP:
            raise ValueError(
                f"Unknown interpolation '{self.interpolation}'. "
                f"Valid options: {sorted(_INTERP_MAP)}."
            )

    @property
    def cv2_interpolation(self) -> int:
        """Return the cv2 integer constant for the configured method."""
        return _INTERP_MAP[self.interpolation]

    @classmethod
    def from_dict(cls, cfg: dict) -> "UpsampleConfig":
        """Construct from a plain dictionary (e.g., parsed YAML section).

        Args:
            cfg: Dictionary with keys matching the dataclass field names.

        Returns:
            A populated :class:`UpsampleConfig` instance.
        """
        return cls(
            tile_size=cfg.get("tile_size", 320),
            target_size=cfg.get("target_size", 640),
            interpolation=cfg.get("interpolation", "INTER_CUBIC"),
            overwrite=cfg.get("overwrite", True),
            max_workers=cfg.get("max_workers"),
        )


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable → defined at module scope)
# ---------------------------------------------------------------------------

def _resize_worker(
    src_path_str: str,
    target_size: int,
    tile_size: int,
    interp: int,
    overwrite: bool,
) -> Tuple[str, bool, str]:
    """Resize a single GeoTIFF tile and update its spatial metadata.

    This function runs inside a worker process spawned by
    :class:`~concurrent.futures.ProcessPoolExecutor`.  It is intentionally
    defined at module level so it can be serialised by ``pickle``.

    Processing steps:
        1. Open the source GeoTIFF and read the RGB data plus metadata.
        2. Resize with ``cv2.resize`` using the configured interpolation.
        3. Recompute the Affine transform for the new pixel resolution.
        4. Write the result as a GeoTIFF (preserving CRS and TIFF tags).

    Args:
        src_path_str: Absolute path to the source GeoTIFF (str for pickling).
        target_size: Target width and height in pixels.
        tile_size: Source tile size; used to derive the Affine scale factor.
        interp: cv2 interpolation constant (int).
        overwrite: If ``True``, overwrite the source file; otherwise write
            a new file with a ``_<target_size>`` suffix.

    Returns:
        A ``(path_str, success, error_message)`` tuple.  ``error_message``
        is an empty string on success.
    """
    src_path = Path(src_path_str)

    try:
        # ------------------------------------------------------------------
        # 1. Read source GeoTIFF.
        # ------------------------------------------------------------------
        with rasterio.open(src_path) as ds:
            data      = ds.read()                   # (3, H, W) uint8
            profile   = ds.profile.copy()
            old_tf    = ds.transform
            tags      = ds.tags()
            src_h, src_w = ds.height, ds.width

        # Sanity check: skip already-upsampled files to support reruns.
        if src_h == target_size and src_w == target_size:
            return src_path_str, True, "already target size — skipped"

        # ------------------------------------------------------------------
        # 2. Resize with cv2.
        #    cv2.resize expects HWC layout.
        # ------------------------------------------------------------------
        img_hwc     = data.transpose(1, 2, 0)       # (H, W, 3)
        resized_hwc = cv2.resize(
            img_hwc,
            (target_size, target_size),
            interpolation=interp,
        )
        resized_chw = resized_hwc.transpose(2, 0, 1)  # (3, H, W)

        # ------------------------------------------------------------------
        # 3. Update Affine transform.
        #
        #    The new transform must map the same geographic extent onto
        #    more pixels.  Composing with Affine.scale(src/tgt) shrinks
        #    the per-pixel ground footprint proportionally:
        #
        #        new_tf * (new_col, new_row)
        #      = old_tf * (new_col × scale, new_row × scale)
        #      = old_tf * (old_col, old_row)     [same ground position]
        # ------------------------------------------------------------------
        scale      = tile_size / target_size
        new_tf     = old_tf * Affine.scale(scale)

        # ------------------------------------------------------------------
        # 4. Build output profile and destination path.
        # ------------------------------------------------------------------
        profile.update(
            width=target_size,
            height=target_size,
            transform=new_tf,
            dtype="uint8",
            count=3,
        )

        if overwrite:
            dst_path = src_path
        else:
            dst_path = src_path.with_name(
                f"{src_path.stem}_{target_size}{src_path.suffix}"
            )

        # Write to a temp file first when overwriting to protect against
        # partial writes corrupting the source.
        tmp_path = dst_path.with_suffix(".tmp.tif")
        with rasterio.open(tmp_path, "w", **profile) as ds_out:
            ds_out.write(resized_chw)
            # Re-attach all original TIFF tags and add the new resolution tag.
            ds_out.update_tags(**tags, upsampled_size=str(target_size))

        tmp_path.replace(dst_path)

        return src_path_str, True, ""

    except Exception as exc:  # noqa: BLE001
        return src_path_str, False, str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ImageUpsampler:
    """Upsamples all GeoTIFF tiles in the specified splits.

    Uses :class:`~concurrent.futures.ProcessPoolExecutor` for parallelism.
    Spatial metadata (CRS, Affine transform) is preserved and updated to
    reflect the new pixel resolution.

    Args:
        config: :class:`UpsampleConfig` with all upsampling hyperparameters.
    """

    def __init__(self, config: UpsampleConfig) -> None:
        self._cfg = config

    def upsample_splits(self, images_dir: Path, splits: List[str]) -> None:
        """Upsample all tiles in the listed splits.

        Args:
            images_dir: Root directory containing ``<split>/``
                subdirectories with GeoTIFF tiles.
            splits: List of split names to process (e.g.,
                ``["train", "val"]``).

        Raises:
            FileNotFoundError: If *images_dir* does not exist.
        """
        if not images_dir.exists():
            raise FileNotFoundError(f"images_dir does not exist: {images_dir}")

        cfg     = self._cfg
        workers = cfg.max_workers or os.cpu_count() or 1

        logger.info(
            "Upsampling: %dpx → %dpx  interpolation=%s  overwrite=%s  workers=%d",
            cfg.tile_size, cfg.target_size, cfg.interpolation,
            cfg.overwrite, workers,
        )

        # Collect all tile paths across requested splits.
        all_paths: List[Path] = []
        for split in splits:
            split_dir = images_dir / split
            if not split_dir.exists():
                logger.warning(
                    "Upsample: images/%s/ does not exist — skipping.", split
                )
                continue
            tiles = sorted(split_dir.glob("*.tif"))
            logger.info("  %-5s: %d tile(s)  (%s)", split, len(tiles), split_dir)
            all_paths.extend(tiles)

        if not all_paths:
            logger.error("No tiles found in the specified splits. Aborting upsample.")
            return

        total = len(all_paths)
        logger.info("Total tiles to process: %d", total)

        task_args = [
            (str(p), cfg.target_size, cfg.tile_size, cfg.cv2_interpolation, cfg.overwrite)
            for p in all_paths
        ]

        # ------------------------------------------------------------------
        # Parallel processing with progress tracking.
        # ------------------------------------------------------------------
        t0         = time.perf_counter()
        done       = succeeded = failed = 0
        errors: List[Tuple[str, str]] = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_resize_worker, *args): args[0]
                for args in task_args
            }

            for future in as_completed(future_map):
                path_str, ok, msg = future.result()
                done += 1

                if ok:
                    succeeded += 1
                    if msg:  # e.g. "already target size"
                        logger.debug("%s: %s", Path(path_str).name, msg)
                else:
                    failed += 1
                    errors.append((path_str, msg))
                    logger.warning("FAILED: %s — %s", Path(path_str).name, msg)

                # Log progress every 5 % or at completion.
                if done % max(1, total // 20) == 0 or done == total:
                    elapsed = time.perf_counter() - t0
                    eta     = (elapsed / done) * (total - done)
                    logger.info(
                        "  Progress: %d/%d (%.1f%%)  elapsed=%.0fs  ETA=%.0fs",
                        done, total, done / total * 100, elapsed, eta,
                    )

        elapsed_total = time.perf_counter() - t0
        rate = elapsed_total / total * 1000.0 if total > 0 else 0.0

        logger.info(
            "Upsample complete.  succeeded=%d  failed=%d  "
            "total_time=%.1fs  rate=%.1fms/image",
            succeeded, failed, elapsed_total, rate,
        )

        if errors:
            logger.error("Errors (%d):", len(errors))
            for path_str, msg in errors[:20]:
                logger.error("  %s: %s", Path(path_str).name, msg)
            if len(errors) > 20:
                logger.error("  … and %d more.", len(errors) - 20)

        logger.info(
            "Labels (.txt) were NOT modified — YOLO OBB coordinates are "
            "normalised [0, 1] fractions and are scale-invariant."
        )
