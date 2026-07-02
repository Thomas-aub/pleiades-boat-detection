"""
pansharpen_pipeline.py — Highly Optimized Multiprocessed Pansharpening
======================================================================
Single entry point for dataset pansharpening. Traverses all subdirectories
to find Multispectral (MS) and Panchromatic (PAN) image pairs, aligns them,
and applies a pansharpening algorithm (e.g., Brovey).

Architecture
------------
- Single File: All logic and parameters are contained within this script.
- Memory Safe: Uses rasterio windowed I/O and WarpedVRT to process images
  block-by-block. RAM usage remains flat regardless of image size.
- Multiprocessed: Uses ProcessPoolExecutor to process multiple image pairs
  concurrently across available CPU cores.

Dependencies
------------
    pip install numpy rasterio tqdm
"""

from __future__ import annotations

import logging
import concurrent.futures
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm


# =============================================================================
# ⚙️ CONFIGURATION (Set all parameters here)
# =============================================================================

@dataclass(frozen=True)
class Config:
    # --- Paths ---
    input_root: Path = Path("data/inf")
    output_root: Path = Path("data/pansharpened")
    
    # --- File Discovery (Pléiades NEO Specific) ---
    valid_extensions: Tuple[str, ...] = (".tif", ".tiff", ".TIF", ".TIFF")
    exclude_tag: str = "_NED_"
    ms_require_tag: str = "_RGB_"
    
    # --- Pansharpening Parameters ---
    method: str = "brovey"
    resample_algo: str = "cubic"
    output_dtype: str = "uint16"
    compress: str = "lzw"
    block_size: int = 512
    
    # --- Execution ---
    max_workers: int = 8
    overwrite: bool = False
    log_level: str = "INFO"


CONFIG = Config()


# =============================================================================
# Logging Setup
# =============================================================================

def configure_logger(level: str) -> logging.Logger:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("pansharpening")
    logger.setLevel(numeric_level)
    if not logger.handlers:
        console = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(message)s", 
            datefmt="%H:%M:%S"
        )
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

logger = configure_logger(CONFIG.log_level)


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def discover_pairs(input_root: Path, cfg: Config) -> List[Tuple[Path, Path]]:
    """
    Recursively discover MS and PAN pairs in the input directory,
    handling the specific Pléiades NEO folder and naming structures.
    """
    pairs = []
    logger.info("Scanning %s for image pairs...", input_root)
    
    # 1. Grab exclusively TIF files (ignoring .TFW, .XML, .JPG, etc.)
    all_tifs = [
        f for f in input_root.rglob("*") 
        if f.is_file() and f.suffix.upper() in (".TIF", ".TIFF")
    ]
    
    # 2. Filter for purely MS RGB images (exclude _NED_ files)
    ms_files = [
        f for f in all_tifs 
        if cfg.ms_require_tag in f.name and cfg.exclude_tag not in f.name
    ]
    
    for ms_path in ms_files:
        # 3. Translate MS path to PAN path
        # Example MS Dir : .../000373521_2_1_STD_A/IMG_01_PNEO4_MS-FS
        # Example PAN Dir: .../000373521_2_1_STD_A/IMG_01_PNEO4_PAN
        pan_dir_str = str(ms_path.parent).replace("_MS-FS", "_PAN")
        pan_dir = Path(pan_dir_str)
        
        # Example MS Name : IMG_PNEO4_STD_2023..._MS-FS_ORT_..._F_1_RGB_R2C2.TIF
        # Example PAN Name: IMG_PNEO4_STD_2023..._PAN_ORT_..._F_1_P_R2C2.TIF
        pan_name = ms_path.name.replace("_MS-FS_", "_PAN_").replace("_RGB_", "_P_")
        
        pan_path = pan_dir / pan_name
        
        # 4. Validate the pair exists
        if pan_path.exists():
            pairs.append((ms_path, pan_path))
        else:
            logger.warning("Found MS image but missing PAN counterpart:\n  MS:  %s\n  PAN: %s", 
                           ms_path, pan_path)
            
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Core Pansharpening Math
# ---------------------------------------------------------------------------

def _apply_brovey_block(
    ms_block: np.ndarray, 
    pan_block: np.ndarray, 
    dtype: np.dtype
) -> np.ndarray:
    """
    Apply Brovey Transform to a single block.
    Formula: PS = (MS / mean(MS)) * PAN
    """
    # Convert to float32 for arithmetic precision
    ms_float = ms_block.astype(np.float32)
    pan_float = pan_block.astype(np.float32)
    
    # Calculate pseudo-pan (mean of MS bands)
    pseudo_pan = np.mean(ms_float, axis=0)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(pseudo_pan > 0, pan_float / pseudo_pan, 0.0)
    
    # Apply ratio
    ps_float = ms_float * ratio
    
    # Clip and cast to output dtype
    dtype_max = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    ps_clipped = np.clip(ps_float, 0, dtype_max)
    
    return ps_clipped.astype(dtype)


# ---------------------------------------------------------------------------
# Single Pair Processing (Worker Function)
# ---------------------------------------------------------------------------

def process_single_pair(ms_path: Path, pan_path: Path, cfg: Config) -> Tuple[bool, str]:
    """
    Pansharpen a single pair using windowed I/O to maintain memory safety.
    Designed to run isolated in a separate process.
    """
    try:
        # Calculate relative path and rename the "MS-FS" folder to "PANSHARP"
        rel_path_str = str(ms_path.parent.relative_to(cfg.input_root))
        rel_path_clean = Path(rel_path_str.replace("_MS-FS", "_PANSHARP"))
        
        out_dir = cfg.output_root / rel_path_clean
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Rename output file to reflect pansharpening
        out_name = ms_path.name.replace("_MS-FS_", "_PANSHARP_").replace("_RGB_", "_PS_")
        out_path = out_dir / out_name
        
        if out_path.exists() and not cfg.overwrite:
            return True, f"Skipped (already exists): {out_name}"

        resampling_map = {
            "nearest": Resampling.nearest,
            "bilinear": Resampling.bilinear,
            "cubic": Resampling.cubic,
            "lanczos": Resampling.lanczos
        }
        resample_enum = resampling_map.get(cfg.resample_algo, Resampling.bilinear)

        with rasterio.open(pan_path) as pan_src:
            with rasterio.open(ms_path) as ms_src:
                
                # Setup output profile matching PAN dimensions but MS band count
                profile = pan_src.profile.copy()
                profile.update(
                    dtype=cfg.output_dtype,
                    count=ms_src.count,
                    compress=cfg.compress,
                    predictor=2,
                    tiled=True,
                    blockxsize=cfg.block_size,
                    blockysize=cfg.block_size,
                    BIGTIFF="YES"
                )
                
                # Configure WarpedVRT to dynamically upsample MS to PAN resolution and alignment
                vrt_options = {
                    'resampling': resample_enum,
                    'transform': pan_src.transform,
                    'height': pan_src.height,
                    'width': pan_src.width,
                    'crs': pan_src.crs
                }
                
                # Use a temporary file for safe writing
                tmp_path = out_path.with_suffix(".tmp.tif")
                
                with WarpedVRT(ms_src, **vrt_options) as vrt_ms:
                    with rasterio.open(tmp_path, "w", **profile) as dst:
                        
                        # Process block-by-block
                        for _, window in dst.block_windows(1):
                            # Read window from high-res PAN (1 band)
                            pan_data = pan_src.read(1, window=window)
                            
                            # Read window from low-res MS (dynamically upsampled by VRT)
                            ms_data = vrt_ms.read(window=window)
                            
                            # Skip entirely empty blocks
                            if pan_data.max() == 0 and ms_data.max() == 0:
                                dst.write(np.zeros_like(ms_data, dtype=cfg.output_dtype), window=window)
                                continue
                                
                            # Apply pansharpening
                            if cfg.method.lower() == "brovey":
                                ps_data = _apply_brovey_block(ms_data, pan_data, np.dtype(cfg.output_dtype))
                            else:
                                raise ValueError(f"Unknown method: {cfg.method}")
                                
                            dst.write(ps_data, window=window)
                            
                        # Carry over relevant tags
                        dst.update_tags(**ms_src.tags(), pansharpened="true", method=cfg.method)
                
                # Atomic swap
                tmp_path.replace(out_path)
                return True, f"Success: {out_name}"

    except Exception as exc:
        return False, f"Failed {ms_path.name}: {str(exc)}"


# ---------------------------------------------------------------------------
# Multiprocessing Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("═" * 62)
    logger.info("  Starting Multiprocessed Pansharpening Pipeline")
    logger.info("═" * 62)
    
    start_time = time.perf_counter()
    
    # Ensure roots exist
    CONFIG.input_root.mkdir(parents=True, exist_ok=True)
    CONFIG.output_root.mkdir(parents=True, exist_ok=True)
    
    # 1. Discover Data
    pairs = discover_pairs(CONFIG.input_root, CONFIG)
    if not pairs:
        logger.error("No image pairs found in %s.", CONFIG.input_root.resolve())
        sys.exit(0)
        
    logger.info("Found %d pair(s) to process.", len(pairs))
    logger.info("Config: max_workers=%d, method=%s, compress=%s, overwrite=%s", 
                CONFIG.max_workers, CONFIG.method, CONFIG.compress, CONFIG.overwrite)

    # 2. Parallel Execution
    success_count = 0
    fail_count = 0
    
    logger.info("Dispatching tasks to ProcessPoolExecutor...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG.max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_pair, ms_path, pan_path, CONFIG): ms_path 
            for ms_path, pan_path in pairs
        }
        
        # Track completion with progress bar
        with tqdm(total=len(pairs), desc="Pansharpening", unit="pair") as pbar:
            for future in concurrent.futures.as_completed(futures):
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.error(message)
                pbar.update(1)

    # 3. Summary
    elapsed = time.perf_counter() - start_time
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    time_str = f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{s}s")

    logger.info("═" * 62)
    logger.info("  Pipeline Complete in %s", time_str)
    logger.info("  ✓ %d Successful  |  ✗ %d Failed  |  Total %d", 
                success_count, fail_count, len(pairs))
    logger.info("═" * 62)


if __name__ == "__main__":
    main()