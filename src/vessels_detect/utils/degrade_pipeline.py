"""
=============================================================================
Forward Physical Degradation Pipeline: Pléiades Neo → Pléiades HR Simulation
=============================================================================
This pipeline implements a physics-grounded, sensor-matched degradation chain
for simulating Pléiades HR (0.50 m pansharpened product) imagery from
Pléiades Neo (0.30 m native) input. 

THEORETICAL BASIS & SENSOR DIFFERENCES:
-----------------------------------------
Naively downsampling 0.30 m to 0.50 m is insufficient. The two sensors 
differ across four independent physical dimensions:

  1. Spectral Response Functions (SRFs): Pléiades HR carries 4 wide MS bands 
     while Neo carries 6 narrower bands [5][6]. The `spectral_misalign` step 
     simulates HR colorimetry via linear mixing matrices derived from 
     spectral overlap integrals.
  2. PSF / MTF: The HR system MTF at Nyquist (0.70 m) is measured at 0.16 [1][2], 
     while Neo is significantly sharper (>0.22) [3]. The `mtf_blur` step applies 
     a Gaussian kernel calibrated to match the delivered HR product MTF [9].
  3. Native vs. Product GSD: HR products (0.50 m) are upsampled from a native 
     0.70 m detector pitch [5]. The `resize` stage replicates this Airbus 
     ground-processing via a two-step decimation/resampling sequence.
  4. Signal-to-Noise Ratio (SNR): HR SNR at 100 W·m⁻²·µm⁻¹·sr⁻¹ is ≈150 [1][4]. 
     Because spatial decimation mathematically cleans the image, `sensor_noise` 
     injects calibrated Poisson–Gaussian noise to reconstruct the HR noise floor.

REFERENCES:
-----------
  [1] Kubik, P. et al., "PLEIADES-HR Radiometric Image Quality", ISPRS, 2012.
  [2] Lebègue, L. et al., "Star-Based Methods for Pléiades HR", ISPRS, 2012.
  [3] Cantrell, S.J. et al., "Pléiades Neo Imager", USGS OFR 2021-1030-P, 2024.
  [4] Martin, V. et al., "PLEIADES-HR 1A&1B Image Quality", SPIE 8866, 2013.
  [5] Airbus Intelligence, "Pléiades Imagery User Guide", 2019.
  [6] Airbus Intelligence, "Pléiades Neo User Guide", October 2021.
  [9] Blanchet, G. et al., "Measuring the MTF of PLEIADES-HR", GRETSI, 2017.

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
    sys.exit("ERROR: 'tqdm' is required.  pip install tqdm")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window
    from rasterio.transform import Affine
except ImportError:
    sys.exit("ERROR: 'rasterio' is required.  pip install rasterio")


# ── Folders ───────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "data/raw"
OUTPUT_FOLDER = "data/raw50"

# ── Hardware & Memory ─────────────────────────────────────────────────────────
GPU_ENABLED  = True
GPU_DEVICE   = 0
TILE_SIZE    = 4096
TILE_OVERLAP = 256

# ── Spectral Response Function (SRF) Matrices ─────────────────────────────────
# Maps Neo input bands to HR output bands to simulate older colorimetry.
# Leave default for accurate physics. Tweak weights manually to 
# "warm up" or "cool down" specific colors (e.g., lower Blue to reduce haze).


SPECTRAL_MATRICES = {
    # 6-band Neo (DB, B, G, R, RE, NIR) -> 4-band HR (B, G, R, NIR)
    6: [
        [0.050, 0.900, 0.050, 0.000, 0.000, 0.000],  
        [0.000, 0.100, 0.900, 0.000, 0.000, 0.000],  
        [0.000, 0.000, 0.000, 0.950, 0.050, 0.000],  
        [0.000, 0.000, 0.000, 0.000, 0.050, 0.950],  
    ],
    # 4-band Neo (B, G, R, NIR) -> 4-band HR (B, G, R, NIR)
    4: [
        [0.900, 0.100, 0.000, 0.000],  
        [0.150, 0.850, 0.000, 0.000],  
        [0.000, 0.000, 1.000, 0.000],  
        [0.000, 0.000, 0.000, 1.000],  
    ],
    # 3-band Neo RGB (B, G, R) -> 3-band HR RGB (B, G, R)
    3: [
        [0.900, 0.100, 0.000],
        [0.150, 0.850, 0.000],
        [0.000, 0.000, 1.000],
    ]
}

# ── Physics Pipeline ──────────────────────────────────────────────────────────
PIPELINE = [
    {
        "op": "spectral_misalign",
        # Auto-selects matrix based on input band count.
        # To disable color shifting entirely, pass an empty dict: {}
        "matrices": SPECTRAL_MATRICES,
    },
    {
        "op": "mtf_blur",
        # Simulates Modulation Transfer Function (MTF) of the Pléiades HR sensor system
        # - mtf_nyquist: Sharpness. 
        #       Lower = blurry raw physics. 
        #       Higher = sharper, simulating a restored ground-station product.
        # - kernel_size: Blur radius. Keep odd. Use 11-15 to prevent halo ringing.
        "op": "mtf_blur",
        "mtf_nyquist_x": 0.35,
        "mtf_nyquist_y": 0.35,
        "target_freq": 0.5 * (0.30 / 0.50),
        "kernel_size": 11,
    },
    
    {
        "op": "resize",
        # Simulates hardware detector grid. Do not alter scales.
        "scale": 7.0 / 3.0,   # 0.30 m → 0.70 m 
        "resampling": "cubic",
    },
    {
        "op": "resize",
        # Simulates Airbus upsampling. 
        "scale": 5.0 / 7.0,   # 0.70 m → 0.50 m
        "resampling": "cubic",
    },
    {
        "op": "sensor_noise",
        # Simulates Signal-to-Noise Ratio (SNR) limitations by injecting physical photon grain and electronic static
        # - photon_peak: Shot noise. Lower (<8000) = grainy. Higher (>15000) = clean.
        # - read_noise_sigma: Electronic static in dark areas. 
        "photon_peak": 15000.0,
        "read_noise_sigma": 0.0001,
        "seed": 42,
    },
]


# ── Output Configuration ──────────────────────────────────────────────────────
OUTPUT_DTYPE   = None       # None → preserve source dtype
COMPRESS       = "lzw"
OVERWRITE      = False
TIF_EXTENSIONS = (".tif", ".TIF", ".tiff", ".TIFF")
LOG_LEVEL      = "INFO"

# =============================================================================
# INTERNAL IMPLEMENTATION
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def _init_backend(enabled: bool, device: int) -> Any:
    """Return CuPy if available and enabled, else NumPy."""
    if not enabled:
        log.info("Backend : NumPy  (GPU_ENABLED=False)")
        return np
    try:
        import cupy as cp
        cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        log.info("Backend : CuPy   [device %d — %s]", device, name)
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
# Spatial & Tiling
# ---------------------------------------------------------------------------

@dataclass
class SpatialState:
    """Current raster dimensions and georeferencing transform."""
    width:     int
    height:    int
    transform: Affine
    crs:       Any


@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    overlap:   int

    @property
    def stride(self) -> int:
        return max(1, self.tile_size - self.overlap)


@dataclass(frozen=True)
class Tile:
    read_window:   Window
    src_transform: Affine
    write_window:  Window


def iter_tiles(width: int, height: int, transform: Affine, cfg: TileConfig):
    """Yield tiles covering the full raster with symmetric overlap padding.

    Only the central ``write_window`` is committed to disk; the overlap absorbs
    edge artefacts from operations with large spatial support (e.g. FFT blur).
    """
    for y0 in range(0, height, cfg.stride):
        for x0 in range(0, width, cfg.stride):
            core_w = min(cfg.stride, width  - x0)
            core_h = min(cfg.stride, height - y0)
            pad_left   = min(x0,              cfg.overlap // 2)
            pad_top    = min(y0,              cfg.overlap // 2)
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


def output_window(tile: Tile, scale: float, out_width: int = 0, out_height: int = 0) -> Window:
    """Map the source write-window to the destination pixel grid.

    ``out_width`` / ``out_height`` clamp the result to prevent off-by-one
    overflows caused by ``round()`` accumulation on the last tile.
    """
    w     = tile.write_window
    out_x = round(w.col_off / scale)
    out_y = round(w.row_off  / scale)
    out_w = round((w.col_off + w.width)  / scale) - out_x
    out_h = round((w.row_off + w.height) / scale) - out_y
    if out_width  > 0:
        out_w = min(out_w, out_width  - out_x)
    if out_height > 0:
        out_h = min(out_h, out_height - out_y)
    return Window(out_x, out_y, max(1, out_w), max(1, out_h))


def crop_tile(
    band: np.ndarray,
    tile: Tile,
    scale: float,
    out_width: int = 0,
    out_height: int = 0,
) -> np.ndarray:
    """Crop a processed band to the central write region at output resolution."""
    col_off = round((tile.write_window.col_off - tile.read_window.col_off) / scale)
    row_off = round((tile.write_window.row_off - tile.read_window.row_off) / scale)
    ow = output_window(tile, scale, out_width, out_height)
    return band[row_off : row_off + ow.height, col_off : col_off + ow.width]


# ---------------------------------------------------------------------------
# Physics Operations
# ---------------------------------------------------------------------------

def _build_mtf_kernel(
    kernel_size: int,
    mtf_nyquist_x: float,
    mtf_nyquist_y: float,
    target_freq: float = 0.5,
) -> np.ndarray:
    """Build a normalised 2-D Gaussian kernel matched to the target sensor MTF.

    Solves MTF(f) = exp(-2π²σ²f²) for σ at (target_freq, mtf_nyquist):

        σ = sqrt(-ln(mtf_nyquist) / (2π² · target_freq²))

    Set target_freq = 0.5 × (GSD_src / GSD_dst) to calibrate against the
    destination sampling grid rather than the source Nyquist [9].
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd; got {kernel_size}.")

    def _sigma(mtf: float) -> float:
        safe = max(1e-9, min(mtf, 1.0 - 1e-9))
        return math.sqrt(-math.log(safe) / (2.0 * math.pi**2 * target_freq**2))

    sigma_x, sigma_y = _sigma(mtf_nyquist_x), _sigma(mtf_nyquist_y)
    log.debug("MTF kernel: σ_x=%.3f  σ_y=%.3f  (target_freq=%.4f)", sigma_x, sigma_y, target_freq)

    t = kernel_size // 2
    xx, yy = np.meshgrid(np.arange(-t, t + 1, dtype=np.float64),
                         np.arange(-t, t + 1, dtype=np.float64))
    kernel = np.exp(-((xx**2) / (2.0 * sigma_x**2) + (yy**2) / (2.0 * sigma_y**2)))
    return kernel / np.sum(kernel)


def op_spectral_misalign(
    bands: List[Any],
    nodata_masks: List[Any],
    step: Dict,
) -> Tuple[List[Any], List[Any]]:
    """Remap Neo bands to HR colorimetry via a linear SRF mixing matrix.

    M (n_out × n_in): L_HR[i] = Σ_j M[i,j] · L_Neo[j].
    Resolves matrix from ``step["mixing_matrix"]`` or ``step["matrices"][n_bands]``.
    """
    mixing_matrix = step.get("mixing_matrix") or step.get("matrices", {}).get(len(bands))
    if mixing_matrix is None:
        log.debug("spectral_misalign: no matrix for %d-band input — passthrough.", len(bands))
        return bands, nodata_masks

    M = xp.asarray(np.array(mixing_matrix, dtype=np.float64))  # (n_out, n_in)
    n_out, n_in = int(M.shape[0]), int(M.shape[1])

    if len(bands) != n_in:
        raise ValueError(f"spectral_misalign: matrix expects {n_in} bands, got {len(bands)}.")

    H, W = bands[0].shape
    remapped = (M @ xp.stack(bands, axis=0).reshape(n_in, H * W)).reshape(n_out, H, W)

    out_bands: List[Any] = []
    out_masks: List[Any] = []
    for i in range(n_out):
        # Mask pixels where any significant contributor (weight > 0.05) is nodata.
        mask_out = xp.zeros((H, W), dtype=bool)
        for j in range(n_in):
            if float(M[i, j]) > 0.05:
                mask_out = mask_out | nodata_masks[j]
        out_bands.append(xp.where(~mask_out, remapped[i], xp.float64(0.0)))
        out_masks.append(mask_out)

    return out_bands, out_masks


def op_mtf_blur(
    bands: List[Any],
    nodata_masks: List[Any],
    step: Dict,
) -> List[Any]:
    """Convolve each band with the HR sensor PSF in the Fourier domain.

    Nodata pixels are excluded via normalised weighted averaging to prevent
    boundary bleed.
    """
    kernel_cpu = _build_mtf_kernel(
        step["kernel_size"],
        step["mtf_nyquist_x"],
        step["mtf_nyquist_y"],
        target_freq=step.get("target_freq", 0.5),
    )
    # Zero-centre the kernel before FFT to avoid a circular phase shift.
    H_fft = _to_device(
        np.fft.fft2(np.fft.ifftshift(kernel_cpu), s=bands[0].shape).astype(np.complex128)
    )

    out_bands = []
    for band, mask in zip(bands, nodata_masks):
        valid   = (~mask).astype(xp.float64)
        filled  = xp.where(~mask, band, xp.float64(0.0))
        blurred = xp.real(xp.fft.ifft2(xp.fft.fft2(filled) * H_fft))
        weights = xp.real(xp.fft.ifft2(xp.fft.fft2(valid)  * H_fft))
        out_bands.append(xp.where(weights > 1e-6, blurred / weights, band))
    return out_bands


def op_sensor_noise(
    bands: List[Any],
    nodata_masks: List[Any],
    tile_rng: np.random.Generator,
    step: Dict,
) -> List[Any]:
    """Inject Poisson-Gaussian noise to match the HR sensor noise floor.

    σ²_total = σ²_shot + σ²_read,  where σ_shot = sqrt(x / photon_peak).
    Must run after all resize steps so noise variance is at the correct scale.
    """
    photon_peak      = float(step["photon_peak"])
    read_noise_sigma = float(step["read_noise_sigma"])

    out_bands = []
    for band, mask in zip(bands, nodata_masks):
        shot_sigma = xp.sqrt(xp.clip(band, 0.0, 1.0) * photon_peak) / photon_peak
        shot_noise = _to_device(tile_rng.normal(0.0, _to_host(shot_sigma), band.shape))
        read_noise = _to_device(tile_rng.normal(0.0, read_noise_sigma,     band.shape))
        noisy = xp.clip(band + shot_noise + read_noise, 0.0, 1.0)
        out_bands.append(xp.where(~mask, noisy, band))
    return out_bands


def op_resize(
    bands: List[Any],
    masks: List[Any],
    state: SpatialState,
    step: Dict,
) -> Tuple[List[Any], List[Any], SpatialState]:
    """Resample bands to a new GSD via rasterio reprojection.

    scale > 1 → downsampling (0.30 → 0.70 m).
    scale < 1 → upsampling   (0.70 → 0.50 m).
    """
    scale          = float(step["scale"])
    resampling_alg = getattr(Resampling, step.get("resampling", "cubic"))

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
            source=arr_cpu,                destination=out,
            src_transform=state.transform, src_crs=state.crs,
            dst_transform=dst_t,           dst_crs=state.crs,
            resampling=interp,
        )
        return out

    out_bands = [_to_device(_reproj(_to_host(b), resampling_alg)) for b in bands]
    out_masks = [
        _to_device(_reproj(_to_host(m).astype(np.float64), Resampling.nearest) > 0.5)
        for m in masks
    ]
    return out_bands, out_masks, new_state


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def run_pipeline(
    bands: List[np.ndarray],
    nodata: Optional[float],
    state: SpatialState,
    pipeline: List[Dict],
    tile_seed: Optional[int] = None,
) -> Tuple[List[np.ndarray], SpatialState]:
    """Run the degradation pipeline on a single tile.

    Band count may change mid-pipeline (e.g. 6-band Neo → 4-band HR after
    spectral_misalign).
    """
    dev_bands = [_to_device(b) for b in bands]
    dev_masks = [
        _to_device(np.isclose(b, nodata)) if nodata is not None
        else _to_device(np.zeros(b.shape, dtype=bool))
        for b in bands
    ]
    tile_rng = np.random.default_rng(tile_seed)

    for step in pipeline:
        op = step["op"]
        if op == "spectral_misalign":
            dev_bands, dev_masks = op_spectral_misalign(dev_bands, dev_masks, step)
        elif op == "mtf_blur":
            dev_bands = op_mtf_blur(dev_bands, dev_masks, step)
        elif op == "sensor_noise":
            dev_bands = op_sensor_noise(dev_bands, dev_masks, tile_rng, step)
        elif op == "resize":
            dev_bands, dev_masks, state = op_resize(dev_bands, dev_masks, state, step)
        else:
            warnings.warn(f"Unknown pipeline op '{op}' — skipped.", stacklevel=2)

    return [_to_host(b) for b in dev_bands], state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_output_band_count(n_input_bands: int, pipeline: List[Dict]) -> int:
    """Return the final band count after all spectral_misalign steps."""
    count = n_input_bands
    for step in pipeline:
        if step["op"] != "spectral_misalign":
            continue
        matrix = step.get("mixing_matrix") or step.get("matrices", {}).get(count)
        if matrix is not None:
            count = len(matrix)
    return count


def _dtype_range(dtype_str: str) -> float:
    """Return the maximum representable value for a numeric dtype."""
    dt = np.dtype(dtype_str)
    return float(np.iinfo(dt).max) if np.issubdtype(dt, np.integer) else 1.0


# ---------------------------------------------------------------------------
# Per-File Processing
# ---------------------------------------------------------------------------

def process_image(
    src_path: Path,
    input_root: Path,
    out_root: Path,
    out_dtype: Optional[str],
    compress: str,
    tile_cfg: TileConfig,
    image_seed: Optional[int],
) -> bool:
    """Degrade a single GeoTIFF and write the result into the output tree."""
    out_path = out_root / src_path.relative_to(input_root)
    if out_path.exists() and not OVERWRITE:
        tqdm.write(f"  ⟳  Already exists, skipping: {src_path.name}")
        return False

    spatial_scale = math.prod(
        float(s["scale"]) for s in PIPELINE if s["op"] == "resize"
    )

    try:
        with rasterio.open(src_path) as src:
            src_meta    = src.meta.copy()
            data_max    = _dtype_range(src.dtypes[0])
            n_out_bands = _infer_output_band_count(src.count, PIPELINE)

            out_w = max(1, int(src.width  / spatial_scale))
            out_h = max(1, int(src.height / spatial_scale))
            out_t = rasterio.transform.from_origin(
                src.transform.c, src.transform.f,
                abs(src.transform.a) * spatial_scale,
                abs(src.transform.e) * spatial_scale,
            )

            eff_dtype  = out_dtype or src.dtypes[0]
            is_int     = np.issubdtype(np.dtype(eff_dtype), np.integer)
            dtype_info = np.iinfo(np.dtype(eff_dtype)) if is_int else None

            def _denorm_and_cast(arr: np.ndarray) -> np.ndarray:
                arr = arr * data_max
                if dtype_info:
                    arr = np.clip(arr, dtype_info.min, dtype_info.max)
                return arr.astype(eff_dtype)

            src_meta.update(
                width=out_w, height=out_h, transform=out_t,
                dtype=eff_dtype, count=n_out_bands,
                compress=compress if compress.lower() != "none" else None,
                BIGTIFF="IF_SAFER",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            n_tiles = (
                math.ceil(src.width  / tile_cfg.stride)
                * math.ceil(src.height / tile_cfg.stride)
            )

            with rasterio.open(out_path, "w", **src_meta) as dst:
                for tile_idx, tile in enumerate(
                    tqdm(
                        iter_tiles(src.width, src.height, src.transform, tile_cfg),
                        total=n_tiles, desc="  tiles", leave=False,
                    )
                ):
                    raw_bands = [
                        src.read(i, window=tile.read_window).astype(np.float64)
                        for i in range(1, src.count + 1)
                    ]
                    norm_nodata = src.nodata / data_max if src.nodata is not None else None
                    tile_state  = SpatialState(
                        width=tile.read_window.width,
                        height=tile.read_window.height,
                        transform=tile.src_transform,
                        crs=src.crs,
                    )
                    tile_bands, _ = run_pipeline(
                        [b / data_max for b in raw_bands],
                        norm_nodata, tile_state, PIPELINE,
                        tile_seed=(image_seed * 65537 + tile_idx) if image_seed else None,
                    )

                    actual_win = output_window(tile, spatial_scale, out_w, out_h)
                    for band_idx, band in enumerate(tile_bands, start=1):
                        dst.write(
                            _denorm_and_cast(crop_tile(band, tile, spatial_scale, out_w, out_h)),
                            band_idx,
                            window=actual_win,
                        )

        tqdm.write(f"    → {out_w}×{out_h}px  ({n_out_bands}-band)  ✓")

        geojson_src = src_path.with_suffix(".geojson")
        if geojson_src.exists():
            shutil.copy2(geojson_src, out_path.with_suffix(".geojson"))

        return True

    except Exception as exc:
        tqdm.write(f"  ✗  FAILED — {src_path.name}: {exc}")
        if out_path.exists():
            out_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    in_root  = Path(INPUT_FOLDER).resolve()
    out_root = Path(OUTPUT_FOLDER).resolve()
    if not in_root.exists() or in_root == out_root:
        sys.exit("ERROR: Check INPUT_FOLDER / OUTPUT_FOLDER paths.")

    tif_files = sorted(
        f for f in in_root.rglob("*")
        if f.is_file() and f.suffix in TIF_EXTENSIONS
    )
    log.info(
        "Found %d file(s).  Pipeline: %s",
        len(tif_files),
        " → ".join(s["op"] for s in PIPELINE),
    )

    success = sum(
        1
        for idx, f in enumerate(tqdm(tif_files, desc="Images"))
        if process_image(
            f, in_root, out_root,
            OUTPUT_DTYPE, COMPRESS,
            TileConfig(TILE_SIZE, TILE_OVERLAP),
            idx,
        )
    )
    log.info("Finished: %d / %d successful.", success, len(tif_files))


if __name__ == "__main__":
    main()