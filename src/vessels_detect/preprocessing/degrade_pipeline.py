"""
=============================================================================
Sensor Degradation Pipeline for Pansharpened GeoTIFF Imagery  [GPU + Tiling]
=============================================================================
Core degradation logic.  Can be run standalone (main) or imported as a
library by build_dataset.py:

    from degrade_pipeline import run_pipeline, SpatialState

Output files mirror INPUT_FOLDER's subfolder structure under OUTPUT_FOLDER,
keeping original filenames.  When run standalone, degraded full-resolution
images are written (no gamma / uint8 conversion — that is build_dataset.py's
responsibility).

=============================================================================
PARAMETERS  –  edit this block
=============================================================================
"""

# ── Folders ───────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "data/eval/raw"
OUTPUT_FOLDER = "data/eval/raw50"

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_ENABLED = True
GPU_DEVICE  = 0

# ── Tiling (For Memory/GPU Chunking, independent of YOLO dataset tiling) ──────
TILE_SIZE    = 4096   # stride + overlap  (peak mem ≈ (tile_size+overlap)² × bands × 8 B)
TILE_OVERLAP = 64

# ── Degradation pipeline ──────────────────────────────────────────────────────
PIPELINE = [
    {"op": "mtf_blur", "mtf_nyquist_x": 0.75, "mtf_nyquist_y": 0.75},

    {"op": "spectral_misalign",
     "global_shift_px": [0.3, 0.2],
     "per_band_sigma_px": 0.1,
     "seed": 42},

    # Pléiades-Neo (30cm) -> Pléiades (50cm) requires a float scale of 50/30
    {"op": "downsample", "scale": 5.0 / 3.0, "resampling": "lanczos"},

    # Signal-dependent noise model parameters (tune based on bit-depth)
    {"op": "add_noise", "gain": 0.05, "read_noise": 0.5, "seed": 42},
]

# ── Output options (standalone mode only) ─────────────────────────────────────
OUTPUT_DTYPE   = None
COMPRESS       = "none"
OVERWRITE      = False
TIF_EXTENSIONS = (".tif", ".TIF", ".tiff", ".TIFF")

LOG_LEVEL = "INFO"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: 'tqdm' is required.  pip install tqdm")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window
    from rasterio.transform import Affine
except ImportError:
    sys.exit("ERROR: 'rasterio' is required.  pip install rasterio")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory Chunking Logic (Embedded)
# ---------------------------------------------------------------------------

@dataclass
class TileConfig:
    tile_size: int
    overlap: int
    @property
    def stride(self) -> int:
        return max(1, self.tile_size - self.overlap)

@dataclass
class Tile:
    read_window: Window
    src_transform: Affine
    write_window: Window

def tile_count(width: int, height: int, cfg: TileConfig) -> int:
    return math.ceil(width / cfg.stride) * math.ceil(height / cfg.stride)

def iter_tiles(width: int, height: int, transform: Affine, cfg: TileConfig):
    for y0 in range(0, height, cfg.stride):
        for x0 in range(0, width, cfg.stride):
            core_w = min(cfg.stride, width - x0)
            core_h = min(cfg.stride, height - y0)
            
            pad_left = min(x0, cfg.overlap // 2)
            pad_top = min(y0, cfg.overlap // 2)
            pad_right = min(width - (x0 + core_w), cfg.overlap - pad_left)
            pad_bottom = min(height - (y0 + core_h), cfg.overlap - pad_top)
            
            read_x = x0 - pad_left
            read_y = y0 - pad_top
            read_w = pad_left + core_w + pad_right
            read_h = pad_top + core_h + pad_bottom
            
            read_win = Window(read_x, read_y, read_w, read_h)
            tile_transform = rasterio.windows.transform(read_win, transform)
            
            yield Tile(
                read_window=read_win,
                src_transform=tile_transform,
                write_window=Window(x0, y0, core_w, core_h)
            )

def output_window(tile: Tile, scale: float) -> Window:
    """Projects the valid input core to the scaled output grid ensuring zero gaps."""
    in_win = tile.write_window
    out_x = int(in_win.col_off / scale)
    out_y = int(in_win.row_off / scale)
    out_w = int((in_win.col_off + in_win.width) / scale) - out_x
    out_h = int((in_win.row_off + in_win.height) / scale) - out_y
    return Window(out_x, out_y, out_w, out_h)

def crop_tile(band: np.ndarray, tile: Tile, scale: float) -> np.ndarray:
    """Crops the overlap out of the degraded array, isolating the valid core."""
    in_col_off = tile.write_window.col_off - tile.read_window.col_off
    in_row_off = tile.write_window.row_off - tile.read_window.row_off
    
    out_col_off = int(in_col_off / scale)
    out_row_off = int(in_row_off / scale)
    
    out_win = output_window(tile, scale)
    return band[out_row_off : out_row_off + out_win.height, 
                out_col_off : out_col_off + out_win.width]

# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------

def _init_backend(enabled: bool, device: int) -> Any:
    if not enabled:
        log.info("Backend  : NumPy (GPU_ENABLED=False)")
        return np
    try:
        import cupy as cp
        cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(device)
        name  = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        log.info("Backend  : CuPy  [device %d — %s]", device, name)
        return cp
    except Exception as exc:
        log.warning("CuPy unavailable (%s) — falling back to NumPy.", exc)
        return np

xp = _init_backend(GPU_ENABLED, GPU_DEVICE)

def _to_device(arr: np.ndarray) -> Any:
    return xp.asarray(arr) if xp is not np else arr

def _to_host(arr: Any) -> np.ndarray:
    return xp.asnumpy(arr) if xp is not np else arr

# ---------------------------------------------------------------------------
# Spatial state
# ---------------------------------------------------------------------------

@dataclass
class SpatialState:
    width:     int
    height:    int
    transform: Any
    crs:       Any

RESAMPLE_MAP = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
    "lanczos":  Resampling.lanczos,
    "average":  Resampling.average,
}

# ---------------------------------------------------------------------------
# Frequency-domain primitives
# ---------------------------------------------------------------------------

def _freq_grids(height: int, width: int) -> Tuple[Any, Any]:
    return xp.meshgrid(xp.fft.fftfreq(height), xp.fft.fftfreq(width), indexing="ij")

def _fft_phase_shift(band: Any, dy: float, dx: float) -> Any:
    Fy, Fx = _freq_grids(*band.shape)
    phase  = xp.exp(xp.asarray(-2j * math.pi) * (Fy * dy + Fx * dx))
    return xp.real(xp.fft.ifft2(xp.fft.fft2(band) * phase))

def _normalised_fft_convolve(band: Any, nodata_mask: Any, H: Any) -> Any:
    valid  = (~nodata_mask).astype(xp.float64)
    filled = xp.where(~nodata_mask, band, xp.float64(0.0))
    bd = xp.real(xp.fft.ifft2(xp.fft.fft2(filled) * H))
    bw = xp.real(xp.fft.ifft2(xp.fft.fft2(valid)  * H))
    return xp.where(bw > 1e-6, bd / bw, band)

# ---------------------------------------------------------------------------
# Pipeline operations — bandwise
# ---------------------------------------------------------------------------

def op_mtf_blur(band: Any, nodata_mask: Any, mtf_nyquist_x: float, mtf_nyquist_y: float) -> Any:
    def _s2(q: float) -> float:
        return -4.0 * math.log(max(1e-9, min(float(q), 1.0 - 1e-9))) / math.pi**2

    Fy, Fx = _freq_grids(*band.shape)
    H = (
        xp.exp(xp.float64(-_s2(mtf_nyquist_x) * math.pi**2) * Fx**2)
        * xp.exp(xp.float64(-_s2(mtf_nyquist_y) * math.pi**2) * Fy**2)
    )
    return _normalised_fft_convolve(band, nodata_mask, H)

def op_add_noise(band: Any, nodata_mask: Any, gain: float, read_noise: float, rng: np.random.Generator) -> Any:
    signal = xp.clip(band, 0.0, None)
    variance = (signal * gain) + (read_noise ** 2)
    std_dev = xp.sqrt(variance)
    noise = _to_device(rng.normal(0.0, _to_host(std_dev), band.shape))
    result = band + noise
    result[nodata_mask] = band[nodata_mask]
    return result

# ---------------------------------------------------------------------------
# Pipeline operations — stack-level
# ---------------------------------------------------------------------------

def op_downsample(
    bands: List[Any], masks: List[Any], state: SpatialState,
    scale: float, resampling: str,
) -> Tuple[List[Any], List[Any], SpatialState]:
    algo = RESAMPLE_MAP.get(resampling.lower())
    if algo is None:
        raise ValueError(f"Unknown resampling '{resampling}'. Choose from: {list(RESAMPLE_MAP)}")

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
            dst_transform=dst_t,           dst_crs=state.crs,
            resampling=interp,
        )
        return out

    out_bands, out_masks = [], []
    for band, mask in zip(bands, masks):
        out_bands.append(_to_device(_reproj(_to_host(band), algo)))
        out_masks.append(_to_device(
            _reproj(_to_host(mask).astype(np.float64), Resampling.nearest) > 0.5
        ))
    return out_bands, out_masks, new_state

def op_spectral_misalign(
    bands: List[Any], masks: List[Any],
    global_shift_px: List[float], per_band_sigma_px: float,
    rng: np.random.Generator,
) -> Tuple[List[Any], List[Any]]:
    gdy, gdx = float(global_shift_px[0]), float(global_shift_px[1])
    out_bands, out_masks = [], []
    for i, (band, mask) in enumerate(zip(bands, masks)):
        dy = gdy + rng.normal(0.0, per_band_sigma_px)
        dx = gdx + rng.normal(0.0, per_band_sigma_px)
        out_bands.append(_fft_phase_shift(band, dy, dx))
        out_masks.append(_fft_phase_shift(mask.astype(xp.float64), dy, dx) > 0.5)
    return out_bands, out_masks

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    bands:    List[np.ndarray],
    nodata:   Optional[float],
    state:    SpatialState,
    pipeline: List[Dict],
) -> Tuple[List[np.ndarray], SpatialState]:
    
    dev_bands = [_to_device(b) for b in bands]
    dev_masks = [
        _to_device(b == nodata if nodata is not None else np.zeros(b.shape, dtype=bool))
        for b in bands
    ]

    for step in pipeline:
        name = step["op"]

        if name == "mtf_blur":
            nx, ny    = float(step["mtf_nyquist_x"]), float(step["mtf_nyquist_y"])
            dev_bands = [op_mtf_blur(b, m, nx, ny) for b, m in zip(dev_bands, dev_masks)]
            
        elif name == "downsample":
            dev_bands, dev_masks, state = op_downsample(
                dev_bands, dev_masks, state,
                scale=float(step["scale"]),
                resampling=step.get("resampling", "average"),
            )
            
        elif name == "spectral_misalign":
            dev_bands, dev_masks = op_spectral_misalign(
                dev_bands, dev_masks,
                global_shift_px   = step.get("global_shift_px",   [0.0, 0.0]),
                per_band_sigma_px = float(step.get("per_band_sigma_px", 0.0)),
                rng               = np.random.default_rng(step.get("seed")),
            )

        elif name == "add_noise":
            rng       = np.random.default_rng(step.get("seed"))
            dev_bands = [op_add_noise(b, m, float(step["gain"]), float(step["read_noise"]), rng)
                         for b, m in zip(dev_bands, dev_masks)]

        else:
            raise ValueError(f"Unknown pipeline op '{name}'.")

    return [_to_host(b) for b in dev_bands], state

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def discover_tifs(root: Path) -> List[Path]:
    return [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and f.suffix in TIF_EXTENSIONS
    ]

def build_output_path(src: Path, in_root: Path, out_root: Path) -> Path:
    return out_root / src.relative_to(in_root)

def _pipeline_spatial_scale(pipeline: List[Dict]) -> float:
    scale = 1.0
    for step in pipeline:
        if step["op"] == "downsample":
            scale *= float(step["scale"])
    return scale

# ---------------------------------------------------------------------------
# Per-file entry point
# ---------------------------------------------------------------------------

def process_image(
    src_path: Path, input_root: Path, out_root: Path,
    out_dtype: Optional[str], compress: str, tile_cfg: TileConfig,
) -> Optional[Path]:
    
    out_path = build_output_path(src_path, input_root, out_root)
    if out_path.exists() and not OVERWRITE:
        tqdm.write(f"  ⟳  Already exists, skipping: {src_path.name}")
        return None

    spatial_scale = _pipeline_spatial_scale(PIPELINE)

    try:
        with rasterio.open(src_path) as src:
            src_transform = src.transform
            src_crs       = src.crs
            src_nodata    = src.nodata
            src_dtype     = src.dtypes[0]
            n_bands       = src.count
            src_meta      = src.meta.copy()

            n_tiles    = tile_count(src.width, src.height, tile_cfg)
            out_w      = max(1, int(src.width / spatial_scale))
            out_h      = max(1, int(src.height / spatial_scale))
            out_t      = rasterio.transform.from_origin(
                src_transform.c, src_transform.f,
                abs(src_transform.a) * spatial_scale,
                abs(src_transform.e) * spatial_scale,
            )

            tqdm.write(
                f"  {src_path.name}  [{n_bands}b  "
                f"{src.width}×{src.height}px  →  {n_tiles} tile(s)]"
            )

            eff_dtype  = out_dtype or src_dtype
            is_int     = np.issubdtype(np.dtype(eff_dtype), np.integer)
            dtype_info = np.iinfo(np.dtype(eff_dtype)) if is_int else None

            def _cast(arr):
                if dtype_info:
                    arr = np.clip(arr, dtype_info.min, dtype_info.max)
                return arr.astype(eff_dtype)

            src_meta.update(
                width=out_w, height=out_h, transform=out_t,
                dtype=eff_dtype,
                compress=compress if compress.lower() != "none" else None,
                BIGTIFF="IF_SAFER",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(out_path, "w", **src_meta) as dst:
                for tile in tqdm(
                    iter_tiles(src.width, src.height, src_transform, tile_cfg),
                    total=n_tiles, desc="  tiles", unit="tile",
                    leave=False, colour="green",
                ):
                    tile_bands = [
                        src.read(i, window=tile.read_window).astype(np.float64)
                        for i in range(1, n_bands + 1)
                    ]
                    tile_state = SpatialState(
                        width=tile.read_window.width, height=tile.read_window.height,
                        transform=tile.src_transform, crs=src_crs,
                    )
                    tile_bands, _ = run_pipeline(tile_bands, src_nodata, tile_state, PIPELINE)

                    actual_win = output_window(tile, spatial_scale)
                    for band_idx, band in enumerate(tile_bands, start=1):
                        cropped = crop_tile(band, tile, spatial_scale)
                        dst.write(_cast(cropped), band_idx, window=actual_win)

        tqdm.write(f"    → {out_w}×{out_h}px  ✓")

        # Handle the matching GeoJSON
        geojson_src = src_path.with_suffix(".geojson")
        if geojson_src.exists():
            geojson_dst = out_path.with_suffix(".geojson")
            if not geojson_dst.exists() or OVERWRITE:
                shutil.copy2(geojson_src, geojson_dst)
                tqdm.write(f"    → Copied labels: {geojson_dst.name} ✓")

        return out_path

    except Exception as exc:
        tqdm.write(f"  ✗  FAILED – {src_path.name}: {exc}")
        log.debug("Exception details:", exc_info=True)
        if out_path.exists():
            out_path.unlink()
        return None

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _log_pipeline() -> None:
    steps = "  →  ".join(
        f"{s['op']}({', '.join(f'{k}={v}' for k, v in s.items() if k != 'op')})"
        for s in PIPELINE
    )
    log.info("Pipeline : %s", steps)

def main() -> None:
    input_root  = Path(INPUT_FOLDER).resolve()
    output_root = Path(OUTPUT_FOLDER).resolve()

    if not input_root.exists():
        sys.exit(f"ERROR: INPUT_FOLDER does not exist: {input_root}")
    if input_root == output_root:
        sys.exit("ERROR: INPUT_FOLDER and OUTPUT_FOLDER must differ.")

    tile_cfg = TileConfig(tile_size=TILE_SIZE, overlap=TILE_OVERLAP)

    log.info("Input    : %s", input_root)
    log.info("Output   : %s", output_root)
    _log_pipeline()
    log.info("Chunking : tile_size=%d  overlap=%d  stride=%d",
             tile_cfg.tile_size, tile_cfg.overlap, tile_cfg.stride)
    log.info("Compress : %s  |  Overwrite : %s", COMPRESS, OVERWRITE)

    tif_files = discover_tifs(input_root)
    if not tif_files:
        log.warning("No TIF files found under %s", input_root)
        return

    log.info("Found %d file(s) to process.", len(tif_files))
    success = skipped = 0

    with tqdm(tif_files, desc="Images", unit="img", colour="cyan") as bar:
        for src_path in bar:
            bar.set_postfix_str(src_path.name[:50], refresh=True)
            result = process_image(src_path, input_root, output_root, OUTPUT_DTYPE, COMPRESS, tile_cfg)
            if result:
                success += 1
            else:
                skipped += 1
            bar.set_postfix(done=success, skipped=skipped, refresh=True)

    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"  Finished  │  ✓ {success}  │  ⟳/✗ {skipped}  │  {len(tif_files)} total")
    tqdm.write(f"{'─'*60}")

if __name__ == "__main__":
    main()