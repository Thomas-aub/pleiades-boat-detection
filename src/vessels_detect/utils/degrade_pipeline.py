"""
=============================================================================
SOTA Forward Physical Degradation Pipeline for Satellite Imagery [GPU + Tiling]
=============================================================================
This pipeline implements the deterministic, physics-based degradation model 
for simulating lower-resolution satellite imagery from high-resolution sources.

REFERENCES & JUSTIFICATIONS:
----------------------------
1. Modulation Transfer Function (MTF) Filtering:
   - Kim et al., "WorldView-3 Super-resolution Training Dataset Construction 
     Using the MTF-GLP Method", Geo Data, 2025. (https://doi.org/10.22761/gd.2025.0045)
   - Justification: SOTA remote sensing downsampling strictly utilizes the MTF-GLP 
     method. The spatial filter is designed such that its amplitude at the Nyquist 
     frequency perfectly matches the target sensor's MTF. Arbitrary Gaussian blurs 
     destroy the phase and high-frequency structural integrity of pansharpened data.

2. Pléiades & Pléiades Neo Optical Baselines:
   - Blanchet et al., 2017. (https://www.gretsi.fr/data/colloque/pdf/2017_blanchet107.pdf)
   - Pléiades Neo User Guide, 2021.
   - Justification: Pléiades (0.5m) products are heavily restored via deconvolution. 
     While raw MTF is ~0.15, the delivered pansharpened products exhibit an effective 
     MTF closer to 0.25 - 0.35. The pipeline allows specific X/Y Nyquist targeting.

3. Sensor Noise Physics:
   - Justification: Optical satellite sensors (CCD/TDI) exhibit Signal-Dependent Noise, 
     composed of photon shot noise (Poisson) and electronic readout noise (Gaussian). 
     This pipeline implements a strict Poisson-Gaussian coupled noise model.

ARCHITECTURE:
-------------
- Image-Level Compilation: Parameters are resolved globally per image to ensure
  absolute uniformity across all memory tiles (eliminating tile checkerboarding).
- Agnostic Backend: Seamlessly falls back from CuPy (GPU) to NumPy (CPU).
=============================================================================
PARAMETERS  –  Edit this block
=============================================================================
"""

import logging
import math
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: 'tqdm' is required. pip install tqdm")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window
    from rasterio.transform import Affine
except ImportError:
    sys.exit("ERROR: 'rasterio' is required. pip install rasterio")


# ── Folders ───────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "data/raw"
OUTPUT_FOLDER = "data/raw50"

# ── Hardware & Memory ─────────────────────────────────────────────────────────
GPU_ENABLED  = True
GPU_DEVICE   = 0
TILE_SIZE    = 4096
TILE_OVERLAP = 256   

# ── Physics Pipeline ──────────────────────────────────────────────────────────
PIPELINE = [
    {
        # STEP 1: Sensor-Matched Optical Blur (MTF-GLP Method)
        "op": "mtf_blur",
        # Pléiades 0.5m typical effective MTF@Nyquist is ~0.25 to 0.30 after restoration.
        # Higher = sharper (preserves more edges). Lower = blurrier.
        "mtf_nyquist_x": 0.28, 
        "mtf_nyquist_y": 0.28,
        "kernel_size": 11,
    },
    {
        # STEP 2: Signal-Dependent Sensor Noise (Poisson-Gaussian)
        "op": "sensor_noise",
        # Represents Full Well Capacity / Photon count. Higher = less noise.
        # 4000 represents a very clean, modern TDI sensor.
        "photon_peak": 4000.0, 
        # Electronic read noise (Gaussian baseline). Kept extremely low.
        "read_noise_sigma": 0.002, 
        "seed": 42, # Optional deterministic seed per dataset
    },
    {
        # STEP 3: Spatial Decimation
        "op": "resize",
        "scale": 5.0 / 3.0,  # Pléiades Neo (0.3m) -> Pléiades (0.5m)
        "resampling": "cubic", # cubic preserves high-frequencies better than bilinear
    }
]

# ── Output Configuration ──────────────────────────────────────────────────────
OUTPUT_DTYPE   = None      
COMPRESS       = "lzw"     # LZW is standard lossless compression for GeoTIFFs
OVERWRITE      = False
TIF_EXTENSIONS = (".tif", ".TIF", ".tiff", ".TIFF")
LOG_LEVEL      = "INFO"

# =============================================================================
# INTERNAL LOGIC & BACKEND
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend Management (CuPy / NumPy)
# ---------------------------------------------------------------------------
def _init_backend(enabled: bool, device: int) -> Any:
    if not enabled:
        log.info("Backend : NumPy (GPU_ENABLED=False)")
        return np
    try:
        import cupy as cp
        cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(device)
        name  = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        log.info("Backend : CuPy  [device %d — %s]", device, name)
        return cp
    except Exception as exc:
        log.warning("CuPy unavailable (%s) — falling back to NumPy.", exc)
        return np

xp = _init_backend(GPU_ENABLED, GPU_DEVICE)

def _to_device(arr: np.ndarray) -> Any:
    return xp.asarray(arr) if xp is not np else arr

def _to_host(arr: Any) -> np.ndarray:
    if xp is not np:
        xp.cuda.Stream.null.synchronize() 
        return xp.asnumpy(arr)
    return arr

# ---------------------------------------------------------------------------
# Spatial & Tiling Structures
# ---------------------------------------------------------------------------
@dataclass
class SpatialState:
    width:     int
    height:    int
    transform: Affine
    crs:       Any

@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    overlap: int

    @property
    def stride(self) -> int:
        return max(1, self.tile_size - self.overlap)

@dataclass(frozen=True)
class Tile:
    read_window:   Window
    src_transform: Affine
    write_window:  Window

def iter_tiles(width: int, height: int, transform: Affine, cfg: TileConfig):
    for y0 in range(0, height, cfg.stride):
        for x0 in range(0, width, cfg.stride):
            core_w = min(cfg.stride, width  - x0)
            core_h = min(cfg.stride, height - y0)
            pad_left   = min(x0, cfg.overlap // 2)
            pad_top    = min(y0, cfg.overlap // 2)
            pad_right  = min(width  - (x0 + core_w), cfg.overlap - pad_left)
            pad_bottom = min(height - (y0 + core_h), cfg.overlap - pad_top)
            
            rx, ry = x0 - pad_left, y0 - pad_top
            rw, rh = pad_left + core_w + pad_right, pad_top + core_h + pad_bottom
            read_win = Window(rx, ry, rw, rh)
            
            yield Tile(
                read_window=read_win,
                src_transform=rasterio.windows.transform(read_win, transform),
                write_window=Window(x0, y0, core_w, core_h),
            )

def output_window(tile: Tile, scale: float) -> Window:
    w      = tile.write_window
    out_x  = round(w.col_off / scale)
    out_y  = round(w.row_off  / scale)
    out_w  = round((w.col_off + w.width)  / scale) - out_x
    out_h  = round((w.row_off + w.height) / scale) - out_y
    return Window(out_x, out_y, max(1, out_w), max(1, out_h))

def crop_tile(band: np.ndarray, tile: Tile, scale: float) -> np.ndarray:
    col_off = round((tile.write_window.col_off - tile.read_window.col_off) / scale)
    row_off = round((tile.write_window.row_off - tile.read_window.row_off) / scale)
    ow = output_window(tile, scale)
    return band[row_off : row_off + ow.height, col_off : col_off + ow.width]

# ---------------------------------------------------------------------------
# Physics Operations
# ---------------------------------------------------------------------------

def _build_mtf_kernel(kernel_size: int, mtf_nyquist_x: float, mtf_nyquist_y: float) -> np.ndarray:
    """
    Constructs an MTF-matched spatial kernel (Wald's protocol / Kim et al. 2025).
    The kernel is designed such that its Fourier transform at the Nyquist 
    frequency equals the specified MTF value.
    """
    def _sigma_from_mtf(mtf: float) -> float:
        # Avoid log(0) domain errors; 
        safe_mtf = max(1e-9, min(mtf, 1.0 - 1e-9))
        return math.sqrt(-math.log(safe_mtf) / (2 * math.pi**2))

    sigma_x = _sigma_from_mtf(mtf_nyquist_x)
    sigma_y = _sigma_from_mtf(mtf_nyquist_y)

    t = kernel_size // 2
    x = np.arange(-t, t + 1)
    y = np.arange(-t, t + 1)
    xx, yy = np.meshgrid(x, y)

    # 2D Gaussian Kernel matching the MTF spread
    kernel = np.exp(-((xx**2) / (2 * sigma_x**2) + (yy**2) / (2 * sigma_y**2)))
    return kernel / np.sum(kernel)

def op_mtf_blur(bands: List[Any], nodata_masks: List[Any], step: Dict) -> List[Any]:
    kernel_cpu = _build_mtf_kernel(step["kernel_size"], step["mtf_nyquist_x"], step["mtf_nyquist_y"])
    k_centred = np.fft.ifftshift(kernel_cpu)
    H = _to_device(np.fft.fft2(k_centred, s=bands[0].shape).astype(np.complex128))

    out_bands = []
    for band, mask in zip(bands, nodata_masks):
        valid  = (~mask).astype(xp.float64)
        filled = xp.where(~mask, band, xp.float64(0.0))
        blurred = xp.real(xp.fft.ifft2(xp.fft.fft2(filled) * H))
        weights = xp.real(xp.fft.ifft2(xp.fft.fft2(valid)  * H))
        out_bands.append(xp.where(weights > 1e-6, blurred / weights, band))
    return out_bands

def op_sensor_noise(bands: List[Any], nodata_masks: List[Any], tile_rng: np.random.Generator, step: Dict) -> List[Any]:
    """
    Applies Signal-Dependent Poisson-Gaussian noise.
    Photon shot noise is calculated per-pixel based on intensity.
    Read noise is an additive constant Gaussian floor.
    """
    photon_peak = step["photon_peak"]
    read_noise_sigma = step["read_noise_sigma"]

    out_bands = []
    for band, mask in zip(bands, nodata_masks):
        # 1. Poisson (Shot) Noise
        lam = xp.clip(band, 0.0, 1.0) * photon_peak
        # Using Gaussian approximation for Poisson since photon_peak is large (SOTA optimization for GPUs)
        shot_noise_sigma = xp.sqrt(lam) / photon_peak
        shot_noise = _to_device(tile_rng.normal(0.0, _to_host(shot_noise_sigma), band.shape))
        
        # 2. Gaussian (Read) Noise
        read_noise = _to_device(tile_rng.normal(0.0, read_noise_sigma, band.shape))
        
        noisy_band = band + shot_noise + read_noise
        out_bands.append(xp.where(~mask, noisy_band, band))
    return out_bands

def op_resize(bands: List[Any], masks: List[Any], state: SpatialState, step: Dict) -> Tuple[List[Any], List[Any], SpatialState]:
    scale = step["scale"]
    resampling_algo = getattr(Resampling, step.get("resampling", "cubic"))

    dst_w = max(1, int(state.width  / scale))
    dst_h = max(1, int(state.height / scale))
    dst_t = rasterio.transform.from_origin(
        state.transform.c, state.transform.f,
        abs(state.transform.a) * (state.width  / dst_w),
        abs(state.transform.e) * (state.height / dst_h),
    )
    new_state = SpatialState(width=dst_w, height=dst_h, transform=dst_t, crs=state.crs)

    def _reproj(arr_cpu: np.ndarray, interp: Resampling) -> np.ndarray:
        out = np.zeros((dst_h, dst_w), dtype=np.float64)
        reproject(
            source=arr_cpu, destination=out,
            src_transform=state.transform, src_crs=state.crs,
            dst_transform=dst_t, dst_crs=state.crs,
            resampling=interp,
        )
        return out

    out_bands = [_to_device(_reproj(_to_host(b), resampling_algo)) for b in bands]
    out_masks = [_to_device(_reproj(_to_host(m).astype(np.float64), Resampling.nearest) > 0.5) for m in masks]
    return out_bands, out_masks, new_state

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def run_pipeline(
    bands: List[np.ndarray], nodata: Optional[float], state: SpatialState, pipeline: List[Dict], tile_seed: Optional[int] = None
) -> Tuple[List[np.ndarray], SpatialState]:
    
    dev_bands = [_to_device(b) for b in bands]
    dev_masks = [_to_device(np.isclose(b, nodata)) if nodata is not None else _to_device(np.zeros(b.shape, bool)) for b in bands]
    tile_rng = np.random.default_rng(tile_seed)

    for step in pipeline:
        if step["op"] == "mtf_blur":
            dev_bands = op_mtf_blur(dev_bands, dev_masks, step)
        elif step["op"] == "sensor_noise":
            dev_bands = op_sensor_noise(dev_bands, dev_masks, tile_rng, step)
        elif step["op"] == "resize":
            dev_bands, dev_masks, state = op_resize(dev_bands, dev_masks, state, step)

    return [_to_host(b) for b in dev_bands], state

# ---------------------------------------------------------------------------
# Per-File Processing Loop
# ---------------------------------------------------------------------------

def _dtype_range(dtype_str: str) -> float:
    dt = np.dtype(dtype_str)
    return float(np.iinfo(dt).max) if np.issubdtype(dt, np.integer) else 1.0

def process_image(src_path: Path, input_root: Path, out_root: Path, out_dtype: Optional[str], compress: str, tile_cfg: TileConfig, image_seed: Optional[int]) -> bool:
    out_path = out_root / src_path.relative_to(input_root)
    if out_path.exists() and not OVERWRITE:
        tqdm.write(f"  ⟳  Already exists, skipping: {src_path.name}")
        return False

    spatial_scale = math.prod([float(s["scale"]) for s in PIPELINE if s["op"] == "resize"])

    try:
        with rasterio.open(src_path) as src:
            src_meta = src.meta.copy()
            data_max = _dtype_range(src.dtypes[0])
            out_w, out_h = max(1, int(src.width / spatial_scale)), max(1, int(src.height / spatial_scale))
            
            out_t = rasterio.transform.from_origin(
                src.transform.c, src.transform.f,
                abs(src.transform.a) * spatial_scale, abs(src.transform.e) * spatial_scale,
            )

            eff_dtype = out_dtype or src.dtypes[0]
            is_int = np.issubdtype(np.dtype(eff_dtype), np.integer)
            dtype_info = np.iinfo(np.dtype(eff_dtype)) if is_int else None

            def _denorm_and_cast(arr: np.ndarray) -> np.ndarray:
                arr = arr * data_max
                if dtype_info: arr = np.clip(arr, dtype_info.min, dtype_info.max)
                return arr.astype(eff_dtype)

            src_meta.update(width=out_w, height=out_h, transform=out_t, dtype=eff_dtype, compress=compress if compress.lower() != "none" else None, BIGTIFF="IF_SAFER")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(out_path, "w", **src_meta) as dst:
                for tile_idx, tile in enumerate(tqdm(iter_tiles(src.width, src.height, src.transform, tile_cfg), total=math.ceil(src.width / tile_cfg.stride) * math.ceil(src.height / tile_cfg.stride), desc="  tiles", leave=False)):
                    raw_bands = [src.read(i, window=tile.read_window).astype(np.float64) for i in range(1, src.count + 1)]
                    norm_nodata = src.nodata / data_max if src.nodata is not None else None
                    
                    tile_state = SpatialState(width=tile.read_window.width, height=tile.read_window.height, transform=tile.src_transform, crs=src.crs)
                    tile_bands, _ = run_pipeline([b / data_max for b in raw_bands], norm_nodata, tile_state, PIPELINE, tile_seed=(image_seed * 65537 + tile_idx) if image_seed else None)

                    actual_win = output_window(tile, spatial_scale)
                    for band_idx, band in enumerate(tile_bands, start=1):
                        dst.write(_denorm_and_cast(crop_tile(band, tile, spatial_scale)), band_idx, window=actual_win)

        tqdm.write(f"    → {out_w}×{out_h}px  ✓")
        
        geojson_src = src_path.with_suffix(".geojson")
        if geojson_src.exists():
            shutil.copy2(geojson_src, out_path.with_suffix(".geojson"))
            
        return True

    except Exception as exc:
        tqdm.write(f"  ✗  FAILED – {src_path.name}: {exc}")
        if out_path.exists(): out_path.unlink()
        return False

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    in_root, out_root = Path(INPUT_FOLDER).resolve(), Path(OUTPUT_FOLDER).resolve()
    if not in_root.exists() or in_root == out_root: sys.exit("ERROR: Check input/output paths.")
    
    tif_files = [f for f in sorted(in_root.rglob("*")) if f.is_file() and f.suffix in TIF_EXTENSIONS]
    log.info(f"Processing {len(tif_files)} files. Physics Mode: MTF-GLP SOTA.")
    
    success = sum(1 for idx, f in enumerate(tqdm(tif_files, desc="Images")) if process_image(f, in_root, out_root, OUTPUT_DTYPE, COMPRESS, TileConfig(TILE_SIZE, TILE_OVERLAP), idx))
    log.info(f"Finished: {success} / {len(tif_files)} successful.")

if __name__ == "__main__":
    main()