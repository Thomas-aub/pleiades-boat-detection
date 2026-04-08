"""
=============================================================================
Real-ESRGAN Faithful Degradation Pipeline for GeoTIFF Imagery  [GPU + Tiling]
=============================================================================
Core degradation logic. Can be run standalone (main) or imported as a
library by build_dataset.py:

    from degrade_pipeline import run_pipeline, SpatialState

Implements the degradation model from:
    "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
    Synthetic Data" — Wang et al., ICCVW 2021.
    https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf

DEGRADATION ORDER (second-order by default, each pass = classical model):
    Pass 1:  blur  →  noise  →  resize
    Pass 2:  blur  →  noise  →  resize

Blur choices      : isotropic Gaussian | anisotropic Gaussian |
                    generalized Gaussian (β-shape) | plateau-shaped | sinc
Noise choices     : additive Gaussian (color or gray) | Poisson (color or gray)
Resize choices    : area | bilinear | bicubic  (nearest excluded — misalignment)

(Note: JPEG compression is intentionally omitted — TIF images do not have it.)

YAML-friendly PIPELINE block (edit the parameters block below):

    PIPELINE:
      - op: gaussian_blur
        ...
      - op: gaussian_noise | poisson_noise
        ...
      - op: resize
        ...

Output files mirror INPUT_FOLDER's subfolder structure under OUTPUT_FOLDER,
keeping original filenames.
=============================================================================
PARAMETERS  –  edit this block
=============================================================================
"""

# ── Folders ───────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "data/raw30"
OUTPUT_FOLDER = "data/raw50"

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_ENABLED = True
GPU_DEVICE  = 0

# ── Tiling (memory/GPU chunking, independent of YOLO dataset tiling) ──────────
TILE_SIZE    = 4096   # peak mem ≈ (tile_size + overlap)² × bands × 8 B
TILE_OVERLAP = 256

# ── Degradation pipeline (Real-ESRGAN second-order model) ─────────────────────
#
# Each pass reproduces the classical degradation model: blur → noise → resize.
# Probabilities govern which variant is sampled at runtime; set prob=1.0 to
# always apply a given step. Kernel sizes must be odd integers.
#
# ── Blur ops ──────────────────────────────────────────────────────────────────
#   gaussian_blur
#     sigma_x, sigma_y   : std-dev range [min, max] along each principal axis
#     theta_range        : rotation angle range [min_deg, max_deg]
#                          (set equal for isotropic; use [0,180] for anisotropic)
#     kernel_sizes       : candidate odd kernel sizes to sample from
#     prob               : probability this step is applied
#
#   generalized_gaussian_blur
#     Same as gaussian_blur, plus:
#     beta_range         : shape parameter range [min, max];
#                          β=1 → Laplacian, β=2 → Gaussian, β→∞ → box
#
#   plateau_blur
#     Same as gaussian_blur, plus:
#     beta_range         : shape parameter β; larger β → flatter plateau
#
#   sinc_blur
#     cutoff_range       : normalised cutoff frequency ωc range [min, max]
#                          (units: radians/sample, (0, π])
#     kernel_sizes       : candidate odd kernel sizes
#     prob               : probability this step is applied
#
# ── Noise ops ─────────────────────────────────────────────────────────────────
#   gaussian_noise
#     sigma_range        : noise std-dev range [min, max] in the same units as
#                          the (normalised) image intensities, i.e. [0, 1]
#     gray_prob          : probability of using the same noise on all bands
#                          (gray noise); otherwise independent per-band (color)
#     prob               : probability this step is applied
#
#   poisson_noise
#     scale_range        : Poisson scale λ range [min, max]; scales the noisy
#                          draw before adding it back, i.e. noise ∝ λ·signal
#     gray_prob          : same gray/color toggle as gaussian_noise
#     prob               : probability this step is applied
#
# ── Resize op ─────────────────────────────────────────────────────────────────
#   resize
#     scale              : downsampling factor > 1  (e.g. 5/3 for 30cm→50cm)
#     methods            : candidate methods to sample from uniformly
#                          choices: "area" | "bilinear" | "bicubic"
#                          (nearest-neighbor excluded — introduces misalignment)
#
PIPELINE = [
    # ── Pass 1 ──────────────────────────────────────────────────────────────
    {
        "op": "gaussian_blur",
        "sigma_x":     [0.2, 3.0],  # paper: [0.2, 3] for first pass
        "sigma_y":     [0.2, 3.0],
        "theta_range": [0, 180],    # anisotropic rotation, degrees
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        "prob": 0.7,
    },
    # Generalized Gaussian drawn with probability ~0.15 of total blur budget;
    # set prob to 0 to disable.
    {
        "op": "generalized_gaussian_blur",
        "sigma_x":     [0.2, 3.0],
        "sigma_y":     [0.2, 3.0],
        "theta_range": [0, 180],
        "beta_range":  [0.5, 4.0],  # paper: β ∈ [0.5, 4]
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        "prob": 0.15,
    },
    {
        "op": "plateau_blur",
        "sigma_x":     [0.2, 3.0],
        "sigma_y":     [0.2, 3.0],
        "theta_range": [0, 180],
        "beta_range":  [1.0, 2.0],  # paper: β ∈ [1, 2] for plateau
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        "prob": 0.15,
    },
    {
        "op": "gaussian_noise",
        "sigma_range": [1.0, 30.0],  # paper: sigma ∈ [1, 30]  (uint8 scale)
        "gray_prob":   0.4,           # paper: 40% gray noise
        "prob":        0.5,           # paper: Gaussian vs Poisson = 50/50
        "seed":        42,
    },
    {
        "op": "poisson_noise",
        "scale_range": [0.05, 3.0],  # paper: scale ∈ [0.05, 3]
        "gray_prob":   0.4,
        "prob":        0.5,
        "seed":        42,
    },
    {
        "op": "resize",
        "scale":   5.0 / 3.0,       # Pleiades-Neo 30cm → Pleiades 50cm
        "methods": ["area", "bilinear", "bicubic"],
    },

    # ── Pass 2 ──────────────────────────────────────────────────────────────
    # Paper: second pass uses narrower ranges ([0.2, 1.5] and [1, 25] / [0.05, 2.5])
    # and skips blur with probability 0.2 (controlled via prob).
    {
        "op": "gaussian_blur",
        "sigma_x":     [0.2, 1.5],
        "sigma_y":     [0.2, 1.5],
        "theta_range": [0, 180],
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        "prob": 0.56,               # 0.7 × (1 − 0.2 skip) ≈ 0.56
    },
    {
        "op": "generalized_gaussian_blur",
        "sigma_x":     [0.2, 1.5],
        "sigma_y":     [0.2, 1.5],
        "theta_range": [0, 180],
        "beta_range":  [0.5, 4.0],
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        "prob": 0.12,
    },
    {
        "op": "plateau_blur",
        "sigma_x":     [0.2, 1.5],
        "sigma_y":     [0.2, 1.5],
        "theta_range": [0, 180],
        "beta_range":  [1.0, 2.0],
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        "prob": 0.12,
    },
    {
        "op": "gaussian_noise",
        "sigma_range": [1.0, 25.0],  # paper: tighter range for pass 2
        "gray_prob":   0.4,
        "prob":        0.5,
        "seed":        43,
    },
    {
        "op": "poisson_noise",
        "scale_range": [0.05, 2.5],
        "gray_prob":   0.4,
        "prob":        0.5,
        "seed":        43,
    },
    # Pass 2 resize is a no-op scale=1.0 (no second spatial downsampling for
    # satellite data; remove or set scale > 1.0 to enable a second pass resize).
    {
        "op": "resize",
        "scale":   1.0,
        "methods": ["area", "bilinear", "bicubic"],
    },
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
from scipy.special import j1  # Bessel J₁ for sinc kernel

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
# Memory-chunking helpers
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
            core_w = min(cfg.stride, width  - x0)
            core_h = min(cfg.stride, height - y0)

            pad_left   = min(x0,                  cfg.overlap // 2)
            pad_top    = min(y0,                  cfg.overlap // 2)
            pad_right  = min(width  - (x0 + core_w), cfg.overlap - pad_left)
            pad_bottom = min(height - (y0 + core_h), cfg.overlap - pad_top)

            rx, ry = x0 - pad_left, y0 - pad_top
            rw, rh = pad_left + core_w + pad_right, pad_top + core_h + pad_bottom
            read_win      = Window(rx, ry, rw, rh)
            tile_transform = rasterio.windows.transform(read_win, transform)

            yield Tile(
                read_window=read_win,
                src_transform=tile_transform,
                write_window=Window(x0, y0, core_w, core_h),
            )


def output_window(tile: Tile, scale: float) -> Window:
    """Maps the valid input core to the scaled output grid (no gaps)."""
    w = tile.write_window
    out_x = int(w.col_off / scale)
    out_y = int(w.row_off  / scale)
    out_w = int((w.col_off + w.width)  / scale) - out_x
    out_h = int((w.row_off + w.height) / scale) - out_y
    return Window(out_x, out_y, out_w, out_h)


def crop_tile(band: np.ndarray, tile: Tile, scale: float) -> np.ndarray:
    """Strips the overlap region, leaving only the valid core."""
    col_off = int((tile.write_window.col_off - tile.read_window.col_off) / scale)
    row_off = int((tile.write_window.row_off - tile.read_window.row_off) / scale)
    ow = output_window(tile, scale)
    return band[row_off : row_off + ow.height, col_off : col_off + ow.width]


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


RESAMPLE_MAP: Dict[str, Resampling] = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "bicubic":  Resampling.cubic,
    "area":     Resampling.average,
}

# ---------------------------------------------------------------------------
# Kernel construction (CPU; transferred to device for FFT convolution)
# ---------------------------------------------------------------------------

def _rotation_matrix(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def _anisotropic_gaussian_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta_deg: float,
) -> np.ndarray:
    """Isotropic when sigma_x == sigma_y (theta irrelevant).

    Implements Eq. 2–4 from the paper.
    """
    t = kernel_size // 2
    coords = np.stack(
        np.meshgrid(np.arange(-t, t + 1), np.arange(-t, t + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2).astype(np.float64)  # (K², 2)

    R   = _rotation_matrix(theta_deg)
    Sig = np.diag([sigma_x**2, sigma_y**2])
    cov_inv = R @ np.linalg.inv(Sig) @ R.T

    exponent = -0.5 * np.einsum("ni,ij,nj->n", coords, cov_inv, coords)
    k = np.exp(exponent).reshape(kernel_size, kernel_size)
    return k / k.sum()


def _generalized_gaussian_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta_deg: float,
    beta: float,
) -> np.ndarray:
    """pdf ∝ exp(−½ (Cᵀ Σ⁻¹ C)^β); β=2 recovers the standard Gaussian."""
    t = kernel_size // 2
    coords = np.stack(
        np.meshgrid(np.arange(-t, t + 1), np.arange(-t, t + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2).astype(np.float64)

    R       = _rotation_matrix(theta_deg)
    Sig     = np.diag([sigma_x**2, sigma_y**2])
    cov_inv = R @ np.linalg.inv(Sig) @ R.T

    mahal_sq = np.einsum("ni,ij,nj->n", coords, cov_inv, coords)
    k = np.exp(-0.5 * (mahal_sq ** beta)).reshape(kernel_size, kernel_size)
    return k / k.sum()


def _plateau_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta_deg: float,
    beta: float,
) -> np.ndarray:
    """pdf ∝ 1 / (1 + (Cᵀ Σ⁻¹ C)^β); plateau at centre, heavier tails."""
    t = kernel_size // 2
    coords = np.stack(
        np.meshgrid(np.arange(-t, t + 1), np.arange(-t, t + 1), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2).astype(np.float64)

    R       = _rotation_matrix(theta_deg)
    Sig     = np.diag([sigma_x**2, sigma_y**2])
    cov_inv = R @ np.linalg.inv(Sig) @ R.T

    mahal_sq = np.einsum("ni,ij,nj->n", coords, cov_inv, coords)
    k = (1.0 / (1.0 + mahal_sq ** beta)).reshape(kernel_size, kernel_size)
    return k / k.sum()


def _sinc_kernel(kernel_size: int, omega_c: float) -> np.ndarray:
    """2-D sinc (ideal low-pass) kernel — Eq. 6 of the paper.

    k(i,j) = ωc / (2π √(i²+j²)) · J₁(ωc √(i²+j²)),  with k(0,0) = ωc²/(4π).
    """
    t = kernel_size // 2
    ij = np.stack(
        np.meshgrid(np.arange(-t, t + 1), np.arange(-t, t + 1), indexing="ij"),
        axis=-1,
    ).astype(np.float64)  # (K, K, 2)

    r = np.sqrt(ij[..., 0] ** 2 + ij[..., 1] ** 2)

    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(r == 0.0, omega_c ** 2 / (4.0 * math.pi),
                     omega_c / (2.0 * math.pi * r) * j1(omega_c * r))

    k = np.maximum(k, 0.0)  # sinc can go slightly negative — clip per paper
    return k / k.sum()


# ---------------------------------------------------------------------------
# Frequency-domain blur (band-wise, nodata-aware)
# ---------------------------------------------------------------------------

def _fft_convolve(band: Any, nodata_mask: Any, kernel_cpu: np.ndarray) -> Any:
    """Convolves `band` with `kernel` in the frequency domain.

    Nodata pixels are excluded from both the signal and the weight map so
    that border effects don't bleed across valid/invalid boundaries.
    """
    H = _to_device(
        np.fft.fft2(kernel_cpu, s=band.shape).astype(np.complex128)
    )
    valid  = (~nodata_mask).astype(xp.float64)
    filled = xp.where(~nodata_mask, band, xp.float64(0.0))

    blurred = xp.real(xp.fft.ifft2(xp.fft.fft2(filled) * H))
    weights = xp.real(xp.fft.ifft2(xp.fft.fft2(valid)  * H))
    return xp.where(weights > 1e-6, blurred / weights, band)


# ---------------------------------------------------------------------------
# Pipeline operations
# ---------------------------------------------------------------------------

# ── Blur ──────────────────────────────────────────────────────────────────────

def op_gaussian_blur(
    band: Any,
    nodata_mask: Any,
    rng: np.random.Generator,
    sigma_x: Tuple[float, float],
    sigma_y: Tuple[float, float],
    theta_range: Tuple[float, float],
    kernel_sizes: List[int],
    prob: float,
) -> Any:
    if rng.random() > prob:
        return band
    sx    = float(rng.uniform(*sigma_x))
    sy    = float(rng.uniform(*sigma_y))
    theta = float(rng.uniform(*theta_range))
    ks    = int(rng.choice(kernel_sizes))
    k     = _anisotropic_gaussian_kernel(ks, sx, sy, theta)
    return _fft_convolve(band, nodata_mask, k)


def op_generalized_gaussian_blur(
    band: Any,
    nodata_mask: Any,
    rng: np.random.Generator,
    sigma_x: Tuple[float, float],
    sigma_y: Tuple[float, float],
    theta_range: Tuple[float, float],
    beta_range: Tuple[float, float],
    kernel_sizes: List[int],
    prob: float,
) -> Any:
    if rng.random() > prob:
        return band
    sx    = float(rng.uniform(*sigma_x))
    sy    = float(rng.uniform(*sigma_y))
    theta = float(rng.uniform(*theta_range))
    beta  = float(rng.uniform(*beta_range))
    ks    = int(rng.choice(kernel_sizes))
    k     = _generalized_gaussian_kernel(ks, sx, sy, theta, beta)
    return _fft_convolve(band, nodata_mask, k)


def op_plateau_blur(
    band: Any,
    nodata_mask: Any,
    rng: np.random.Generator,
    sigma_x: Tuple[float, float],
    sigma_y: Tuple[float, float],
    theta_range: Tuple[float, float],
    beta_range: Tuple[float, float],
    kernel_sizes: List[int],
    prob: float,
) -> Any:
    if rng.random() > prob:
        return band
    sx    = float(rng.uniform(*sigma_x))
    sy    = float(rng.uniform(*sigma_y))
    theta = float(rng.uniform(*theta_range))
    beta  = float(rng.uniform(*beta_range))
    ks    = int(rng.choice(kernel_sizes))
    k     = _plateau_kernel(ks, sx, sy, theta, beta)
    return _fft_convolve(band, nodata_mask, k)


def op_sinc_blur(
    band: Any,
    nodata_mask: Any,
    rng: np.random.Generator,
    cutoff_range: Tuple[float, float],
    kernel_sizes: List[int],
    prob: float,
) -> Any:
    if rng.random() > prob:
        return band
    omega_c = float(rng.uniform(*cutoff_range))
    ks      = int(rng.choice(kernel_sizes))
    k       = _sinc_kernel(ks, omega_c)
    return _fft_convolve(band, nodata_mask, k)


# ── Noise ─────────────────────────────────────────────────────────────────────

def _apply_gaussian_noise(
    bands: List[Any],
    nodata_masks: List[Any],
    rng: np.random.Generator,
    sigma: float,
    gray: bool,
) -> List[Any]:
    """Adds AWGN; gray=True reuses the same draw for all bands (gray noise)."""
    ref_shape = bands[0].shape

    if gray:
        # Single draw shared across all bands
        noise_cpu = rng.normal(0.0, sigma, ref_shape)
        shared    = _to_device(noise_cpu)
        return [
            xp.where(~m, b + shared, b)
            for b, m in zip(bands, nodata_masks)
        ]

    return [
        xp.where(~m, b + _to_device(rng.normal(0.0, sigma, ref_shape)), b)
        for b, m in zip(bands, nodata_masks)
    ]


def op_gaussian_noise(
    bands: List[Any],
    nodata_masks: List[Any],
    rng: np.random.Generator,
    sigma_range: Tuple[float, float],
    gray_prob: float,
    prob: float,
) -> List[Any]:
    """Additive Gaussian noise with gray/color toggle (paper §3.1)."""
    if rng.random() > prob:
        return bands

    # sigma_range is in uint8 scale [0,255]; normalise to float image [0,1]
    sigma = float(rng.uniform(*sigma_range)) / 255.0
    gray  = rng.random() < gray_prob
    return _apply_gaussian_noise(bands, nodata_masks, rng, sigma, gray)


def op_poisson_noise(
    bands: List[Any],
    nodata_masks: List[Any],
    rng: np.random.Generator,
    scale_range: Tuple[float, float],
    gray_prob: float,
    prob: float,
) -> List[Any]:
    """Poisson noise: intensity-proportional, independent pixels (paper §3.1).

    We draw a Poisson sample with λ = max(signal, 0) and scale the difference
    by `scale` (analogous to the paper's Poisson scale λ parameter).
    """
    if rng.random() > prob:
        return bands

    scale = float(rng.uniform(*scale_range))
    gray  = rng.random() < gray_prob

    def _poisson_draw(band_cpu: np.ndarray) -> np.ndarray:
        lam     = np.clip(band_cpu, 0.0, None)
        sampled = rng.poisson(lam * 255.0).astype(np.float64) / 255.0
        return (sampled - lam) * scale  # noise component only

    ref_cpu = _to_host(bands[0])
    if gray:
        shared = _to_device(_poisson_draw(ref_cpu))
        return [
            xp.where(~m, b + shared, b)
            for b, m in zip(bands, nodata_masks)
        ]

    return [
        xp.where(~m, b + _to_device(_poisson_draw(_to_host(b))), b)
        for b, m in zip(bands, nodata_masks)
    ]


# ── Resize ────────────────────────────────────────────────────────────────────

def op_resize(
    bands: List[Any],
    masks: List[Any],
    state: SpatialState,
    rng: np.random.Generator,
    scale: float,
    methods: List[str],
) -> Tuple[List[Any], List[Any], SpatialState]:
    """Randomly selects one resize algorithm per call (paper §3.1).

    Nearest-neighbor is excluded because it introduces misalignment artifacts.
    """
    if abs(scale - 1.0) < 1e-6:
        return bands, masks, state

    invalid = [m for m in methods if m not in RESAMPLE_MAP]
    if invalid:
        raise ValueError(
            f"Unknown resize method(s) {invalid}. "
            f"Valid choices: {list(RESAMPLE_MAP)}"
        )

    method = str(rng.choice(methods))
    algo   = RESAMPLE_MAP[method]
    log.debug("resize  method=%s  scale=%.4f", method, scale)

    dst_w = max(1, int(state.width  / scale))
    dst_h = max(1, int(state.height / scale))
    dst_t = rasterio.transform.from_origin(
        state.transform.c, state.transform.f,
        abs(state.transform.a) * (state.width  / dst_w),
        abs(state.transform.e) * (state.height / dst_h),
    )
    new_state = SpatialState(
        width=dst_w, height=dst_h, transform=dst_t, crs=state.crs
    )

    def _reproj(arr_cpu: np.ndarray, interp: Resampling) -> np.ndarray:
        out = np.zeros((dst_h, dst_w), dtype=np.float64)
        reproject(
            source=arr_cpu,    destination=out,
            src_transform=state.transform, src_crs=state.crs,
            dst_transform=dst_t,           dst_crs=state.crs,
            resampling=interp,
        )
        return out

    out_bands = [_to_device(_reproj(_to_host(b), algo))  for b in bands]
    out_masks = [
        _to_device(_reproj(_to_host(m).astype(np.float64), Resampling.nearest) > 0.5)
        for m in masks
    ]
    return out_bands, out_masks, new_state


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    bands:    List[np.ndarray],
    nodata:   Optional[float],
    state:    SpatialState,
    pipeline: List[Dict],
) -> Tuple[List[np.ndarray], SpatialState]:
    """Executes the degradation pipeline on a stack of single-band arrays.

    Args:
        bands:    List of (H, W) float64 arrays, one per spectral band.
        nodata:   Sentinel value marking invalid pixels; None if absent.
        state:    Spatial metadata (size, affine transform, CRS).
        pipeline: Sequence of op dicts (see PIPELINE parameter block).

    Returns:
        Tuple of (degraded_bands, updated_state).
    """
    dev_bands = [_to_device(b) for b in bands]
    dev_masks = [
        _to_device(b == nodata if nodata is not None else np.zeros(b.shape, bool))
        for b in bands
    ]

    for step in pipeline:
        name = step["op"]
        seed = step.get("seed")
        rng  = np.random.default_rng(seed)

        # ── Blur ops ──────────────────────────────────────────────────────
        if name == "gaussian_blur":
            dev_bands = [
                op_gaussian_blur(
                    b, m, rng,
                    sigma_x     = tuple(step["sigma_x"]),
                    sigma_y     = tuple(step["sigma_y"]),
                    theta_range = tuple(step["theta_range"]),
                    kernel_sizes= list(step["kernel_sizes"]),
                    prob        = float(step.get("prob", 1.0)),
                )
                for b, m in zip(dev_bands, dev_masks)
            ]

        elif name == "generalized_gaussian_blur":
            dev_bands = [
                op_generalized_gaussian_blur(
                    b, m, rng,
                    sigma_x     = tuple(step["sigma_x"]),
                    sigma_y     = tuple(step["sigma_y"]),
                    theta_range = tuple(step["theta_range"]),
                    beta_range  = tuple(step["beta_range"]),
                    kernel_sizes= list(step["kernel_sizes"]),
                    prob        = float(step.get("prob", 1.0)),
                )
                for b, m in zip(dev_bands, dev_masks)
            ]

        elif name == "plateau_blur":
            dev_bands = [
                op_plateau_blur(
                    b, m, rng,
                    sigma_x     = tuple(step["sigma_x"]),
                    sigma_y     = tuple(step["sigma_y"]),
                    theta_range = tuple(step["theta_range"]),
                    beta_range  = tuple(step["beta_range"]),
                    kernel_sizes= list(step["kernel_sizes"]),
                    prob        = float(step.get("prob", 1.0)),
                )
                for b, m in zip(dev_bands, dev_masks)
            ]

        elif name == "sinc_blur":
            dev_bands = [
                op_sinc_blur(
                    b, m, rng,
                    cutoff_range = tuple(step["cutoff_range"]),
                    kernel_sizes = list(step["kernel_sizes"]),
                    prob         = float(step.get("prob", 1.0)),
                )
                for b, m in zip(dev_bands, dev_masks)
            ]

        # ── Noise ops ─────────────────────────────────────────────────────
        elif name == "gaussian_noise":
            dev_bands = op_gaussian_noise(
                dev_bands, dev_masks, rng,
                sigma_range = tuple(step["sigma_range"]),
                gray_prob   = float(step.get("gray_prob", 0.4)),
                prob        = float(step.get("prob", 1.0)),
            )

        elif name == "poisson_noise":
            dev_bands = op_poisson_noise(
                dev_bands, dev_masks, rng,
                scale_range = tuple(step["scale_range"]),
                gray_prob   = float(step.get("gray_prob", 0.4)),
                prob        = float(step.get("prob", 1.0)),
            )

        # ── Resize op ─────────────────────────────────────────────────────
        elif name == "resize":
            dev_bands, dev_masks, state = op_resize(
                dev_bands, dev_masks, state, rng,
                scale   = float(step["scale"]),
                methods = list(step.get("methods", ["area", "bilinear", "bicubic"])),
            )

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
        if step["op"] == "resize":
            scale *= float(step["scale"])
    return scale


# ---------------------------------------------------------------------------
# Per-file entry point
# ---------------------------------------------------------------------------

def process_image(
    src_path:   Path,
    input_root: Path,
    out_root:   Path,
    out_dtype:  Optional[str],
    compress:   str,
    tile_cfg:   TileConfig,
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

            n_tiles = tile_count(src.width, src.height, tile_cfg)
            out_w   = max(1, int(src.width  / spatial_scale))
            out_h   = max(1, int(src.height / spatial_scale))
            out_t   = rasterio.transform.from_origin(
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

            def _cast(arr: np.ndarray) -> np.ndarray:
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
                        width=tile.read_window.width,
                        height=tile.read_window.height,
                        transform=tile.src_transform,
                        crs=src_crs,
                    )
                    tile_bands, _ = run_pipeline(
                        tile_bands, src_nodata, tile_state, PIPELINE
                    )

                    actual_win = output_window(tile, spatial_scale)
                    for band_idx, band in enumerate(tile_bands, start=1):
                        cropped = crop_tile(band, tile, spatial_scale)
                        dst.write(_cast(cropped), band_idx, window=actual_win)

        tqdm.write(f"    → {out_w}×{out_h}px  ✓")

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
            result = process_image(
                src_path, input_root, output_root,
                OUTPUT_DTYPE, COMPRESS, tile_cfg,
            )
            if result:
                success += 1
            else:
                skipped += 1
            bar.set_postfix(done=success, skipped=skipped, refresh=True)

    tqdm.write(f"\n{'─'*60}")
    tqdm.write(
        f"  Finished  │  ✓ {success}  │  ⟳/✗ {skipped}  │  {len(tif_files)} total"
    )
    tqdm.write(f"{'─'*60}")


if __name__ == "__main__":
    main()