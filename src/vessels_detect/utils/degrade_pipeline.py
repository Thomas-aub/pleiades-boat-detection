"""
=============================================================================
Real-ESRGAN Faithful Degradation Pipeline for GeoTIFF Imagery  [GPU + Tiling]
=============================================================================
Implements the degradation model from:
    Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
    with Pure Synthetic Data", ICCVW 2021.
    https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.pdf

Sensor-specific blur model informed by:
    Kim et al., "WorldView-3 Super-resolution Training Dataset Construction
    Using the MTF-GLP Method", Geo Data, 2025.
    https://doi.org/10.22761/gd.2025.0045

DEGRADATION ORDER (single-pass; second-order optional via PIPELINE config):
    blur  →  noise  →  resize

Blur choices    : isotropic Gaussian | anisotropic Gaussian |
                  generalized Gaussian (β-shape) | plateau-shaped |
                  sinc / sensor-MTF  ← sampled categorically, never stacked
Noise choices   : additive Gaussian (gray or per-band) |
                  Poisson shot noise (gray or per-band) ← applied exclusively
Resize choices  : area | bilinear | bicubic  (nearest excluded)

JPEG compression is intentionally omitted - TIF products do not carry it.

Usage
-----
Standalone:
    python degrade_pipeline.py

Library:
    from degrade_pipeline import run_pipeline, SpatialState

YAML-equivalent configuration is in the PIPELINE block below.
=============================================================================
PARAMETERS  –  edit this block
=============================================================================
"""

# ── Folders ───────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "data/raw"
OUTPUT_FOLDER = "data/raw50"

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_ENABLED = True
GPU_DEVICE  = 0

# ── Tiling (chunking for large scenes; independent of YOLO tile size) ─────────
TILE_SIZE    = 4096
TILE_OVERLAP = 256   # must be ≥ max_kernel_size // 2  (here 21 // 2 = 10)

# ── Degradation pipeline ──────────────────────────────────────────────────────
#
# Blur section  - ONE type is sampled categorically each call.
#   "blur_type_weights" : categorical probability for each blur type.
#     Keys  : "gaussian" | "generalized_gaussian" | "plateau" | "sinc_mtf"
#     Values: non-negative floats; automatically normalised to sum=1.
#   "prob"              : probability the blur step is applied at all.
#
# Gaussian / generalized-Gaussian / plateau kernels:
#   sigma_x, sigma_y  : std-dev range [min, max] (pixels, source resolution)
#   theta_range       : rotation angle range [min_deg, max_deg]
#                       (equal → isotropic; [0,180] → anisotropic)
#   kernel_sizes      : candidate odd sizes to sample from uniformly
#   beta_range        : shape parameter β for gen-Gaussian and plateau
#
# sinc_mtf kernel  (Kim et al. 2025 recommendation):
#   mtf_nyquist_range : measured MTF at Nyquist [min, max].
#                       For Pleiades Neo → 0.50 m/px simulation the target
#                       sensor (Pleiades-1 / SPOT-7 class) has a measured
#                       MTF@Nyquist of ~0.10–0.15. Widen the range to [0.05,
#                       0.25] for domain randomisation.
#   kernel_sizes      : candidate odd sizes to sample from uniformly
#
# Noise section  - EXCLUSIVE: Gaussian XOR Poisson per call.
#   gaussian_prob     : probability of Gaussian noise (vs. Poisson = 1 − p).
#   sigma_range       : AWGN std-dev range in *normalised* [0,1] image units.
#   poisson_scale_range : Poisson draw scale λ (noise ∝ λ·signal).
#   gray_prob         : probability of same noise across all bands.
#   prob              : probability the noise step is applied at all.
#
# Resize section:
#   scale             : downsampling factor > 1 (5/3 for 30 cm → 50 cm).
#   methods           : candidate algorithms sampled uniformly.
#
PIPELINE = [
    # ── Pass 1 ──────────────────────────────────────────────────────────────
    {
        "op": "blur",
        # Categorical sampling - weights are normalised internally.
        # sinc_mtf is upweighted to reflect measured sensor MTF (Kim et al. 2025).
        "blur_type_weights": {
            "gaussian":              0.50,
            "generalized_gaussian":  0.15,
            "plateau":               0.15,
            "sinc_mtf":              0.20,   # sensor-matched blur
        },
        "sigma_x":    [0.2, 3.0],
        "sigma_y":    [0.2, 3.0],
        "theta_range": [0, 180],
        "beta_range":  [0.5, 4.0],
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        # sinc_mtf: Pleiades-1 / SPOT-7 class MTF@Nyquist ≈ 0.10–0.15;
        # widened to [0.05, 0.25] for domain randomisation.
        "mtf_nyquist_range": [0.05, 0.25],
        "prob": 0.8,
    },
    {
        "op": "noise",
        "gaussian_prob":       0.5,       # Poisson prob = 1 - gaussian_prob
        "sigma_range":         [0.004, 0.118],   # [1/255, 30/255] in [0,1]
        "poisson_scale_range": [0.05, 3.0],
        "gray_prob":           0.4,
        "prob":                0.5,
    },
    {
        "op": "resize",
        "scale":   5.0 / 3.0,            # Pleiades Neo 30 cm → 50 cm
        "methods": ["area", "bilinear", "bicubic"],
    },
    # ── Pass 2 (optional – lighter, post-resize perturbation) ───────────────
    # Tighter blur/noise ranges per original paper §3.1.
    # No second resize: adding a second 5/3 resize has no physical analogue
    # for a single sensor-to-sensor simulation.
    {
        "op": "blur",
        "blur_type_weights": {
            "gaussian":             0.50,
            "generalized_gaussian": 0.15,
            "plateau":              0.15,
            "sinc_mtf":             0.20,
        },
        "sigma_x":    [0.2, 1.5],
        "sigma_y":    [0.2, 1.5],
        "theta_range": [0, 180],
        "beta_range":  [0.5, 4.0],
        "kernel_sizes": [7, 9, 11, 13, 15, 17, 19, 21],
        "mtf_nyquist_range": [0.05, 0.25],
        "prob": 0.56,                    # 0.8 × (1 – 0.3 skip probability)
    },
    {
        "op": "noise",
        "gaussian_prob":       0.5,
        "sigma_range":         [0.004, 0.098],   # [1/255, 25/255] in [0,1]
        "poisson_scale_range": [0.05, 2.5],
        "gray_prob":           0.4,
        "prob":                0.5,
    },
]

# ── Output options (standalone mode only) ─────────────────────────────────────
OUTPUT_DTYPE   = None      # None → preserve source dtype
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
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Tiling helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    overlap: int

    def __post_init__(self) -> None:
        min_overlap = 21 // 2  # half of max kernel size
        if self.overlap < min_overlap:
            warnings.warn(
                f"TILE_OVERLAP={self.overlap} < {min_overlap} (half of max kernel "
                "size). Convolution boundary artefacts may bleed into valid crops.",
                stacklevel=2,
            )

    @property
    def stride(self) -> int:
        return max(1, self.tile_size - self.overlap)


@dataclass(frozen=True)
class Tile:
    read_window:   Window
    src_transform: Affine
    write_window:  Window


def tile_count(width: int, height: int, cfg: TileConfig) -> int:
    return math.ceil(width / cfg.stride) * math.ceil(height / cfg.stride)


def iter_tiles(
    width: int, height: int, transform: Affine, cfg: TileConfig
) -> "Generator[Tile, None, None]":
    for y0 in range(0, height, cfg.stride):
        for x0 in range(0, width, cfg.stride):
            core_w = min(cfg.stride, width  - x0)
            core_h = min(cfg.stride, height - y0)

            pad_left   = min(x0,                      cfg.overlap // 2)
            pad_top    = min(y0,                      cfg.overlap // 2)
            pad_right  = min(width  - (x0 + core_w), cfg.overlap - pad_left)
            pad_bottom = min(height - (y0 + core_h), cfg.overlap - pad_top)

            rx, ry = x0 - pad_left, y0 - pad_top
            rw     = pad_left + core_w + pad_right
            rh     = pad_top  + core_h + pad_bottom

            read_win       = Window(rx, ry, rw, rh)
            tile_transform = rasterio.windows.transform(read_win, transform)

            yield Tile(
                read_window=read_win,
                src_transform=tile_transform,
                write_window=Window(x0, y0, core_w, core_h),
            )


def output_window(tile: Tile, scale: float) -> Window:
    """Maps the valid input core to the scaled output grid without gaps.

    Uses round() instead of int() to avoid 1-pixel gaps / double-writes
    at tile boundaries when scale is non-integer (e.g. 5/3).
    """
    w      = tile.write_window
    out_x  = round(w.col_off / scale)
    out_y  = round(w.row_off  / scale)
    out_w  = round((w.col_off + w.width)  / scale) - out_x
    out_h  = round((w.row_off + w.height) / scale) - out_y
    return Window(out_x, out_y, max(1, out_w), max(1, out_h))


def crop_tile(band: np.ndarray, tile: Tile, scale: float) -> np.ndarray:
    """Strips the overlap halo, leaving only the valid core."""
    col_off = round((tile.write_window.col_off - tile.read_window.col_off) / scale)
    row_off = round((tile.write_window.row_off - tile.read_window.row_off) / scale)
    ow = output_window(tile, scale)
    return band[row_off : row_off + ow.height, col_off : col_off + ow.width]


# ---------------------------------------------------------------------------
# GPU / NumPy backend
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
        log.info("Backend : CuPy  [device %d - %s]", device, name)
        return cp
    except Exception as exc:
        log.warning("CuPy unavailable (%s) - falling back to NumPy.", exc)
        return np


xp = _init_backend(GPU_ENABLED, GPU_DEVICE)


def _to_device(arr: np.ndarray) -> Any:
    return xp.asarray(arr) if xp is not np else arr


def _to_host(arr: Any) -> np.ndarray:
    return xp.asnumpy(arr) if xp is not np else arr  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Spatial state
# ---------------------------------------------------------------------------

@dataclass
class SpatialState:
    """Carries geo-spatial metadata through the pipeline."""
    width:     int
    height:    int
    transform: Affine
    crs:       Any


RESAMPLE_MAP: Dict[str, Resampling] = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "bicubic":  Resampling.cubic,
    "area":     Resampling.average,
}

# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def _rotation_matrix(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]])


def _sample_grid(kernel_size: int) -> np.ndarray:
    """Returns an (K², 2) array of integer coordinate offsets centred at 0."""
    t = kernel_size // 2
    return (
        np.stack(
            np.meshgrid(
                np.arange(-t, t + 1),
                np.arange(-t, t + 1),
                indexing="ij",
            ),
            axis=-1,
        )
        .reshape(-1, 2)
        .astype(np.float64)
    )


def _normalised_kernel(raw: np.ndarray, kernel_size: int) -> np.ndarray:
    k = raw.reshape(kernel_size, kernel_size)
    return k / k.sum()


def _anisotropic_gaussian_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta_deg: float,
) -> np.ndarray:
    """Eq. 2–4 of Wang et al. 2021.  Isotropic when sigma_x == sigma_y."""
    coords  = _sample_grid(kernel_size)
    R       = _rotation_matrix(theta_deg)
    cov_inv = R @ np.diag([1.0 / sigma_x**2, 1.0 / sigma_y**2]) @ R.T
    exp     = -0.5 * np.einsum("ni,ij,nj->n", coords, cov_inv, coords)
    return _normalised_kernel(np.exp(exp), kernel_size)


def _generalized_gaussian_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta_deg: float,
    beta: float,
) -> np.ndarray:
    """pdf ∝ exp(−½ (xᵀΣ⁻¹x)^β); β=2 recovers standard Gaussian."""
    coords   = _sample_grid(kernel_size)
    R        = _rotation_matrix(theta_deg)
    cov_inv  = R @ np.diag([1.0 / sigma_x**2, 1.0 / sigma_y**2]) @ R.T
    mahal_sq = np.einsum("ni,ij,nj->n", coords, cov_inv, coords)
    return _normalised_kernel(np.exp(-0.5 * mahal_sq**beta), kernel_size)


def _plateau_kernel(
    kernel_size: int,
    sigma_x: float,
    sigma_y: float,
    theta_deg: float,
    beta: float,
) -> np.ndarray:
    """pdf ∝ 1/(1 + (xᵀΣ⁻¹x)^β); flat plateau with heavy tails."""
    coords   = _sample_grid(kernel_size)
    R        = _rotation_matrix(theta_deg)
    cov_inv  = R @ np.diag([1.0 / sigma_x**2, 1.0 / sigma_y**2]) @ R.T
    mahal_sq = np.einsum("ni,ij,nj->n", coords, cov_inv, coords)
    return _normalised_kernel(1.0 / (1.0 + mahal_sq**beta), kernel_size)


def _sinc_mtf_kernel(kernel_size: int, mtf_nyquist: float) -> np.ndarray:
    """Sensor-matched sinc kernel derived from measured MTF@Nyquist.

    Converts the MTF@Nyquist value (0 < mtf_nyquist ≤ 1) to a normalised
    cut-off frequency via the approximation used in Kim et al. 2025
    (MTF-GLP method):

        ωc = π · mtf_nyquist^(1/β)  with β ≈ 1.0 (linear MTF model)

    The 2-D kernel follows Eq. 6 of Wang et al. 2021:
        k(i,j) = ωc/(2π r) · J₁(ωc·r),  k(0,0) = ωc²/(4π)
    where r = √(i² + j²).

    Negative sidelobes are clipped to zero (per Real-ESRGAN §3.1).
    """
    # Map measured MTF@Nyquist → normalised cut-off frequency in (0, π].
    omega_c = math.pi * mtf_nyquist

    t  = kernel_size // 2
    ij = (
        np.stack(
            np.meshgrid(
                np.arange(-t, t + 1),
                np.arange(-t, t + 1),
                indexing="ij",
            ),
            axis=-1,
        )
        .astype(np.float64)
    )
    r = np.sqrt(ij[..., 0] ** 2 + ij[..., 1] ** 2)

    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(
            r == 0.0,
            omega_c**2 / (4.0 * math.pi),
            omega_c / (2.0 * math.pi * r) * j1(omega_c * r),
        )

    k = np.maximum(k, 0.0)  # clip negative sidelobes (§3.1)
    k_sum = k.sum()
    if k_sum < 1e-12:
        raise ValueError(
            f"sinc_mtf kernel degenerated (sum≈0) for mtf_nyquist={mtf_nyquist:.3f}. "
            "Check mtf_nyquist_range bounds."
        )
    return k / k_sum


# ---------------------------------------------------------------------------
# Frequency-domain blur (nodata-aware, centred kernel)
# ---------------------------------------------------------------------------

def _fft_convolve(band: Any, nodata_mask: Any, kernel_cpu: np.ndarray) -> Any:
    """Frequency-domain convolution with nodata-aware normalisation.

    The kernel is centred with ifftshift before the forward FFT to eliminate
    the K//2-pixel spatial phase shift introduced by zero-padding at [0,0].
    Nodata pixels are zeroed before convolution and excluded from the
    normalisation weight map to prevent boundary bleeding.
    """
    k_centred = np.fft.ifftshift(kernel_cpu)   # fix phase shift
    H = _to_device(np.fft.fft2(k_centred, s=band.shape).astype(np.complex128))

    valid  = (~nodata_mask).astype(xp.float64)
    filled = xp.where(~nodata_mask, band, xp.float64(0.0))

    blurred = xp.real(xp.fft.ifft2(xp.fft.fft2(filled) * H))
    weights = xp.real(xp.fft.ifft2(xp.fft.fft2(valid)  * H))
    return xp.where(weights > 1e-6, blurred / weights, band)


# ---------------------------------------------------------------------------
# Pipeline operations
# ---------------------------------------------------------------------------

def _sample_blur_kernel(
    rng: np.random.Generator,
    step: Dict,
) -> np.ndarray:
    """Categorically samples ONE blur type, never stacks multiple kernels.

    Respects "blur_type_weights" for a well-calibrated categorical draw.
    This matches the original Real-ESRGAN design intent where blur types
    are mutually exclusive within a single pass.
    """
    weights_dict: Dict[str, float] = step["blur_type_weights"]
    types  = list(weights_dict.keys())
    probs  = np.array([weights_dict[t] for t in types], dtype=np.float64)
    probs /= probs.sum()

    chosen = types[int(rng.choice(len(types), p=probs))]

    ks    = int(rng.choice(step["kernel_sizes"]))
    sx    = float(rng.uniform(*step["sigma_x"]))
    sy    = float(rng.uniform(*step["sigma_y"]))
    theta = float(rng.uniform(*step["theta_range"]))

    if chosen == "gaussian":
        return _anisotropic_gaussian_kernel(ks, sx, sy, theta)
    elif chosen == "generalized_gaussian":
        beta = float(rng.uniform(*step["beta_range"]))
        return _generalized_gaussian_kernel(ks, sx, sy, theta, beta)
    elif chosen == "plateau":
        beta = float(rng.uniform(*step["beta_range"]))
        return _plateau_kernel(ks, sx, sy, theta, beta)
    elif chosen == "sinc_mtf":
        mtf_nyquist = float(rng.uniform(*step["mtf_nyquist_range"]))
        return _sinc_mtf_kernel(ks, mtf_nyquist)
    else:
        raise ValueError(f"Unknown blur type '{chosen}'.")


def op_blur(
    bands: List[Any],
    nodata_masks: List[Any],
    rng: np.random.Generator,
    step: Dict,
) -> List[Any]:
    """Applies one categorically-sampled blur kernel to all bands."""
    if rng.random() > float(step.get("prob", 1.0)):
        return bands
    kernel = _sample_blur_kernel(rng, step)
    return [_fft_convolve(b, m, kernel) for b, m in zip(bands, nodata_masks)]


def op_noise(
    bands: List[Any],
    nodata_masks: List[Any],
    rng: np.random.Generator,
    step: Dict,
) -> List[Any]:
    """Applies either Gaussian AWGN or Poisson shot noise - never both.

    Noise sigma / scale are expressed in normalised [0, 1] image units.
    sigma_range should be set accordingly (e.g. [1/255, 30/255] ≈ [0.004,
    0.118] for uint8-equivalent levels on a [0,1]-normalised image, or
    scaled proportionally for 16-bit imagery normalised to [0, 1]).
    """
    if rng.random() > float(step.get("prob", 1.0)):
        return bands

    gray = rng.random() < float(step.get("gray_prob", 0.4))

    if rng.random() < float(step.get("gaussian_prob", 0.5)):
        # Gaussian AWGN
        sigma = float(rng.uniform(*step["sigma_range"]))
        if gray:
            shared = _to_device(rng.normal(0.0, sigma, bands[0].shape))
            return [xp.where(~m, b + shared, b) for b, m in zip(bands, nodata_masks)]
        return [
            xp.where(~m, b + _to_device(rng.normal(0.0, sigma, b.shape)), b)
            for b, m in zip(bands, nodata_masks)
        ]
    else:
        # Poisson shot noise  (intensity-proportional)
        scale = float(rng.uniform(*step["poisson_scale_range"]))

        def _draw(band_cpu: np.ndarray) -> np.ndarray:
            lam     = np.clip(band_cpu, 0.0, None)
            sampled = rng.poisson(lam).astype(np.float64)
            return (sampled - lam) * scale

        if gray:
            shared = _to_device(_draw(_to_host(bands[0])))
            return [xp.where(~m, b + shared, b) for b, m in zip(bands, nodata_masks)]
        return [
            xp.where(~m, b + _to_device(_draw(_to_host(b))), b)
            for b, m in zip(bands, nodata_masks)
        ]


def op_resize(
    bands: List[Any],
    masks: List[Any],
    state: SpatialState,
    rng: np.random.Generator,
    scale: float,
    methods: Sequence[str],
) -> Tuple[List[Any], List[Any], SpatialState]:
    """Randomly picks one resize algorithm and reprojects all bands.

    Nearest-neighbor is excluded to avoid misalignment artefacts.
    """
    if abs(scale - 1.0) < 1e-6:
        return bands, masks, state

    invalid = [m for m in methods if m not in RESAMPLE_MAP]
    if invalid:
        raise ValueError(
            f"Unknown resize method(s): {invalid}. "
            f"Valid choices: {list(RESAMPLE_MAP)}"
        )

    method = str(rng.choice(list(methods)))
    algo   = RESAMPLE_MAP[method]
    log.debug("resize  method=%s  scale=%.4f", method, scale)

    dst_w = max(1, int(state.width  / scale))
    dst_h = max(1, int(state.height / scale))
    dst_t = rasterio.transform.from_origin(
        state.transform.c,
        state.transform.f,
        abs(state.transform.a) * (state.width  / dst_w),
        abs(state.transform.e) * (state.height / dst_h),
    )
    new_state = SpatialState(
        width=dst_w, height=dst_h, transform=dst_t, crs=state.crs
    )

    def _reproj(arr_cpu: np.ndarray, interp: Resampling) -> np.ndarray:
        out = np.zeros((dst_h, dst_w), dtype=np.float64)
        reproject(
            source=arr_cpu,         destination=out,
            src_transform=state.transform, src_crs=state.crs,
            dst_transform=dst_t,           dst_crs=state.crs,
            resampling=interp,
        )
        return out

    out_bands = [_to_device(_reproj(_to_host(b), algo)) for b in bands]
    out_masks = [
        _to_device(
            _reproj(_to_host(m).astype(np.float64), Resampling.nearest) > 0.5
        )
        for m in masks
    ]
    return out_bands, out_masks, new_state


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    bands:       List[np.ndarray],
    nodata:      Optional[float],
    state:       SpatialState,
    pipeline:    List[Dict],
    image_seed:  Optional[int] = None,
) -> Tuple[List[np.ndarray], SpatialState]:
    """Executes the degradation pipeline on a stack of single-band arrays.

    Args:
        bands:      List of (H, W) float64 arrays, one per spectral band.
                    Values must be normalised to [0, 1] before calling if
                    sigma_range is expressed in [0, 1] units.
        nodata:     Sentinel value for invalid pixels; None if absent.
        state:      Spatial metadata (size, affine transform, CRS).
        pipeline:   Sequence of op dicts (see PIPELINE parameter block).
        image_seed: Per-image seed for full reproducibility. When None, the
                    global NumPy random state is used.

    Returns:
        Tuple of (degraded_bands, updated_state).
    """
    dev_bands = [_to_device(b) for b in bands]
    dev_masks = [
        _to_device(
            (b == nodata) if nodata is not None else np.zeros(b.shape, bool)
        )
        for b in bands
    ]

    # Derive per-step seeds from an image-level seed so that each step is
    # independently reproducible without sharing the same draw sequence.
    base_rng = np.random.default_rng(image_seed)

    for i, step in enumerate(pipeline):
        step_seed = int(base_rng.integers(2**31)) + i
        rng = np.random.default_rng(step_seed)
        name = step["op"]

        if name == "blur":
            dev_bands = op_blur(dev_bands, dev_masks, rng, step)

        elif name == "noise":
            dev_bands = op_noise(dev_bands, dev_masks, rng, step)

        elif name == "resize":
            dev_bands, dev_masks, state = op_resize(
                dev_bands, dev_masks, state, rng,
                scale=float(step["scale"]),
                methods=list(step.get("methods", ["area", "bilinear", "bicubic"])),
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
    """Returns the cumulative spatial downscale factor of all resize ops."""
    scale = 1.0
    for step in pipeline:
        if step["op"] == "resize":
            scale *= float(step["scale"])
    return scale


def _dtype_range(dtype_str: str) -> float:
    """Returns the maximum value for a given numpy dtype string."""
    dt = np.dtype(dtype_str)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return 1.0  # floating-point bands assumed already normalised


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_image(
    src_path:   Path,
    input_root: Path,
    out_root:   Path,
    out_dtype:  Optional[str],
    compress:   str,
    tile_cfg:   TileConfig,
    image_seed: Optional[int],
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

            data_max = _dtype_range(src_dtype)

            n_tiles = tile_count(src.width, src.height, tile_cfg)
            out_w   = max(1, int(src.width  / spatial_scale))
            out_h   = max(1, int(src.height / spatial_scale))
            out_t   = rasterio.transform.from_origin(
                src_transform.c,
                src_transform.f,
                abs(src_transform.a) * spatial_scale,
                abs(src_transform.e) * spatial_scale,
            )

            tqdm.write(
                f"  {src_path.name}  [{n_bands}b  "
                f"{src.width}×{src.height}px  →  {n_tiles} tile(s)  "
                f"dtype={src_dtype}  data_max={data_max:.0f}]"
            )

            eff_dtype  = out_dtype or src_dtype
            is_int     = np.issubdtype(np.dtype(eff_dtype), np.integer)
            dtype_info = np.iinfo(np.dtype(eff_dtype)) if is_int else None

            def _denorm_and_cast(arr: np.ndarray) -> np.ndarray:
                """Restores normalised [0,1] values to output dtype range."""
                arr = arr * data_max
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
                for tile_idx, tile in enumerate(
                    tqdm(
                        iter_tiles(src.width, src.height, src_transform, tile_cfg),
                        total=n_tiles, desc="  tiles", unit="tile",
                        leave=False, colour="green",
                    )
                ):
                    # Normalise to [0, 1] for consistent noise sigma semantics.
                    tile_bands = [
                        src.read(i, window=tile.read_window).astype(np.float64)
                        / data_max
                        for i in range(1, n_bands + 1)
                    ]
                    tile_state = SpatialState(
                        width=tile.read_window.width,
                        height=tile.read_window.height,
                        transform=tile.src_transform,
                        crs=src_crs,
                    )
                    # Combine image-level seed with tile index for uniqueness.
                    tile_seed = (
                        (image_seed * 65537 + tile_idx) if image_seed is not None
                        else None
                    )
                    tile_bands, _ = run_pipeline(
                        tile_bands, src_nodata, tile_state, PIPELINE,
                        image_seed=tile_seed,
                    )

                    actual_win = output_window(tile, spatial_scale)
                    for band_idx, band in enumerate(tile_bands, start=1):
                        cropped = crop_tile(band, tile, spatial_scale)
                        dst.write(
                            _denorm_and_cast(cropped), band_idx, window=actual_win
                        )

        tqdm.write(f"    → {out_w}×{out_h}px  ✓")

        # Copy sidecar GeoJSON labels if present.
        # NOTE: Labels are assumed to be in a projected CRS (metres); no
        # coordinate remapping is needed. Verify this assumption if your
        # GeoJSON stores pixel coordinates - those would require remapping
        # by the scale factor.
        geojson_src = src_path.with_suffix(".geojson")
        if geojson_src.exists():
            geojson_dst = out_path.with_suffix(".geojson")
            if not geojson_dst.exists() or OVERWRITE:
                shutil.copy2(geojson_src, geojson_dst)
                tqdm.write(f"    → Labels copied: {geojson_dst.name} ✓")

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
    log.info(
        "Chunking : tile_size=%d  overlap=%d  stride=%d",
        tile_cfg.tile_size, tile_cfg.overlap, tile_cfg.stride,
    )
    log.info("Compress : %s  |  Overwrite : %s", COMPRESS, OVERWRITE)

    tif_files = discover_tifs(input_root)
    if not tif_files:
        log.warning("No TIF files found under %s", input_root)
        return

    log.info("Found %d file(s) to process.", len(tif_files))
    success = skipped = 0

    with tqdm(tif_files, desc="Images", unit="img", colour="cyan") as bar:
        for file_idx, src_path in enumerate(bar):
            bar.set_postfix_str(src_path.name[:50], refresh=True)
            result = process_image(
                src_path, input_root, output_root,
                OUTPUT_DTYPE, COMPRESS, tile_cfg,
                image_seed=file_idx,
            )
            if result:
                success += 1
            else:
                skipped += 1
            bar.set_postfix(done=success, skipped=skipped, refresh=True)

    tqdm.write(f"\n{'─' * 60}")
    tqdm.write(
        f"  Finished  │  ✓ {success}  │  ⟳/✗ {skipped}  │  {len(tif_files)} total"
    )
    tqdm.write(f"{'─' * 60}")


if __name__ == "__main__":
    main()