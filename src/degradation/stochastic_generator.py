"""
generate_stochastic.py — Stochastic synthetic-domain tile generator.

Reads the mixture CSV produced by mixtures.py and applies a randomised
degradation sequence (B / D / N) to each tile, writing one output GeoTIFF
per row into per-image sub-folders.

Degradation operators (Real-ESRGAN-inspired, remote-sensing calibrated):
    B  — blur    : Matches 'Blur_Type' from CSV {iso, aniso, generalized, sinc}
    D  — downsample : cv2.INTER_AREA (physically correct average-area resampling,
                      handles non-integer GSD ratio 0.3 → 0.5 = 5/3)
    N  — noise   : Matches 'Noise_Type' from CSV {gaussian, poisson}

Order, PSF σ, SNR dB, Blur_Type, and Noise_Type are taken from the CSV row; 
B and N parameters are compensated on-the-fly for whether D has already been 
applied.
"""
from __future__ import annotations

import logging
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blur kernels
# ---------------------------------------------------------------------------

def _isotropic_gaussian(sigma: float, ksize: int) -> np.ndarray:
    """Standard round PSF — models diffraction + isotropic atmosphere."""
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    k = np.outer(g, g)
    return k / k.sum()


def _anisotropic_gaussian(sigma: float, ksize: int, rng: random.Random) -> np.ndarray:
    """Elongated PSF — models along-track jitter / integration smear."""
    sigma_y = sigma * rng.uniform(1.5, 2.5)
    angle   = rng.uniform(0, math.pi)
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    xr =  cos_a * xx + sin_a * yy
    yr = -sin_a * xx + cos_a * yy
    k = np.exp(-0.5 * ((xr / sigma) ** 2 + (yr / sigma_y) ** 2))
    return k / k.sum()


def _generalised_gaussian(sigma: float, ksize: int, rng: random.Random) -> np.ndarray:
    """Generalised (super/sub-Gaussian) PSF via shape parameter β."""
    beta = rng.uniform(0.5, 4.0)
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    r2 = np.add.outer(ax ** 2, ax ** 2)
    k = np.exp(-(r2 / (2 * sigma ** 2)) ** beta)
    return k / k.sum()


def _sinc_2d(sigma: float, ksize: int) -> np.ndarray:
    """2-D Sinc filter — models ringing from resampling / JPEG2000 encoding."""
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    
    # Avoid division by zero at origin
    with np.errstate(invalid="ignore", divide="ignore"):
        sx = np.where(ax == 0, 1.0, np.sin(math.pi * ax / sigma) / (math.pi * ax / sigma))
    
    k = np.outer(sx, sx)
    
    # Create a 1D Kaiser window of the correct size (ksize)
    window_1d = np.kaiser(ksize, 5.0)
    # Convert it into a 2D window
    window_2d = np.outer(window_1d, window_1d)
    
    # Apply the 2D window to limit sidelobes in both directions
    k = k * window_2d
    k = np.abs(k)
    
    return k / k.sum()

# Maps blur-type strings from CSV to kernel factories
_BLUR_FACTORIES: dict[str, Callable] = {
    "iso": lambda s, k, rng: _isotropic_gaussian(s, k),
    "aniso": lambda s, k, rng: _anisotropic_gaussian(s, k, rng),
    "generalized": lambda s, k, rng: _generalised_gaussian(s, k, rng),
    "sinc": lambda s, k, rng: _sinc_2d(s, k),
}


# ---------------------------------------------------------------------------
# Core degradation operators  (C, H, W)  float64 in, float64 out
# ---------------------------------------------------------------------------

def _blur(image: np.ndarray, psf_sigma: float, blur_type: str, rng: random.Random) -> np.ndarray:
    """Apply a specifically requested blur kernel with sigma=psf_sigma (pixels)."""
    ksize = 2 * int(math.ceil(3 * psf_sigma)) + 1  # 3-sigma half-width, always odd
    ksize = max(ksize, 3)
    
    if blur_type not in _BLUR_FACTORIES:
        raise ValueError(f"Unknown blur_type: {blur_type}")
        
    kernel = _BLUR_FACTORIES[blur_type](psf_sigma, ksize, rng)

    out = np.empty_like(image)
    for c in range(image.shape[0]):
        out[c] = fftconvolve(image[c], kernel, mode="same")
    return out


def _noise_additive(image: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """Additive white Gaussian noise at the target SNR."""
    img64 = image.astype(np.float64)
    signal_power = np.mean(img64 ** 2)
    if signal_power <= 0:
        return image
    snr_linear = 10 ** (snr_db / 20)
    noise_std   = math.sqrt(signal_power) / snr_linear
    seed = rng.randint(0, 2**31)
    noise = np.random.default_rng(seed).normal(0.0, noise_std, img64.shape)
    return img64 + noise


def _noise_poisson(image: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    """Multiplicative (signal-dependent) Poisson shot noise at the target SNR."""
    img64 = image.astype(np.float64)
    signal_power = np.mean(img64 ** 2)
    if signal_power <= 0:
        return image
    snr_linear  = 10 ** (snr_db / 20)
    noise_std   = math.sqrt(signal_power) / snr_linear
    mean_intensity = np.mean(img64)
    scale = mean_intensity / (noise_std ** 2 + 1e-10)
    seed = rng.randint(0, 2**31)
    noisy = np.random.default_rng(seed).poisson(
        np.clip(img64 * scale, 0, None)
    ).astype(np.float64)
    return noisy / scale

# Maps noise-type strings from CSV to noise functions
_NOISE_FNS: dict[str, Callable] = {
    "gaussian": _noise_additive,
    "poisson": _noise_poisson
}


def _downsample(image: np.ndarray, source_gsd: float, target_gsd: float) -> np.ndarray:
    """
    Area-average downsample via cv2.INTER_AREA.

    cv2.INTER_AREA computes the weighted mean of source pixels whose areas
    overlap each destination pixel — equivalent to ideal CCD integration and
    correct for non-integer scale factors (e.g. 0.3 → 0.5 = 5/3 ×).
    """
    scale = target_gsd / source_gsd
    if abs(scale - 1.0) < 1e-6:
        return image

    C, H, W = image.shape
    new_H = max(1, int(round(H / scale)))
    new_W = max(1, int(round(W / scale)))

    out = np.empty((C, new_H, new_W), dtype=np.float64)
    for c in range(C):
        # cv2 expects (H, W) and float32
        plane = image[c].astype(np.float32)
        out[c] = cv2.resize(plane, (new_W, new_H), interpolation=cv2.INTER_AREA)
    return out


# ---------------------------------------------------------------------------
# Transform orchestrator
# ---------------------------------------------------------------------------

def apply_stochastic_transform(
    tile: np.ndarray,
    order: str,
    psf_sigma: float,
    snr_db: float,
    source_gsd: float,
    target_gsd: float,
    blur_type: str,
    noise_type: str,
    rng: random.Random,
) -> np.ndarray:
    """
    Apply B/D/N operations in `order`, compensating blur sigma and noise SNR
    for whether D has been applied yet.

    B before D: sigma is scaled by (target_gsd / source_gsd) so the physical
                footprint of the kernel matches the target sensor's MTF.
    N before D: average-pooling by factor k reduces noise std by sqrt(k), which
                boosts SNR by 20·log10(k) dB.  We pre-compensate by subtracting
                that dB gain from the requested SNR so the output SNR is correct.
    """
    img = tile.astype(np.float64)
    ds_factor   = target_gsd / source_gsd   # true (non-integer) ratio
    downsampled = False

    for op in order:
        if op == "B":
            # Blur kernel must cover the right number of *current* pixels
            effective_sigma = (
                psf_sigma                           # post-D: image already at target GSD
                if downsampled
                else psf_sigma * ds_factor          # pre-D: scale up to source-GSD pixels
            )
            img = _blur(img, effective_sigma, blur_type, rng)

        elif op == "N":
            if downsampled:
                effective_snr = snr_db
            else:
                # Average-pooling by ds_factor will reduce noise variance by ds_factor²,
                # boosting SNR by 20·log10(ds_factor) dB.
                effective_snr = snr_db - 20 * math.log10(ds_factor)
                
            if noise_type not in _NOISE_FNS:
                raise ValueError(f"Unknown noise_type: {noise_type}")
                
            noise_fn = _NOISE_FNS[noise_type]
            img = noise_fn(img, effective_snr, rng)

        elif op == "D":
            img = _downsample(img, source_gsd, target_gsd)
            downsampled = True

        else:
            raise ValueError(f"Unknown operation '{op}' in order '{order}'")

    return img


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_tile(source_path: Path, row: pd.Series) -> tuple[np.ndarray, Affine, object]:
    """Extract a geographic tile from a GeoTIFF via bounding-box window."""
    with rasterio.open(source_path) as src:
        window = from_bounds(
            row["min_x"], row["min_y"], row["max_x"], row["max_y"],
            src.transform,
        )
        # MODIFICATION : Forçage de la lecture sur les 3 premières bandes (RGB) uniquement
        data      = src.read(indexes=[1, 2, 3], window=window)
        transform = src.window_transform(window)
        crs       = src.crs
    return data, transform, crs


def _write_tile(
    path: Path,
    image: np.ndarray,
    transform: Affine,
    crs: object,
    orig_dtype: np.dtype,
) -> None:
    """Write a (C, H, W) array as a LZW-compressed GeoTIFF, preserving source dtype."""
    # MODIFICATION : Écrêtage strict à 4095 pour forcer le plafond physique 12-bit
    image_out = np.clip(image, 0, 4095).astype(orig_dtype)

    c, h, w = image_out.shape
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w, count=c,
        dtype=orig_dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(image_out)


# ---------------------------------------------------------------------------
# Worker — processes a single (image_id, group) pair
# ---------------------------------------------------------------------------

def _process_image(image_id: str, group: pd.DataFrame, source_dir: Path, output_dir: Path) -> int:
    """
    Process all mixture rows for one source image.
    Returns the number of tiles successfully written.
    """
    source_path = source_dir / image_id
    if not source_path.exists():
        logger.error("Source not found: %s", source_path)
        return 0

    img_out_dir = output_dir / source_path.stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for _, row in group.iterrows():
        out_path = img_out_dir / f"{source_path.stem}_{row['tile_id']}_{row['Transform_id']}.tif"
        if out_path.exists():
            written += 1
            continue  # idempotent — resume-safe

        try:
            tile, transform, crs = _read_tile(source_path, row)
            orig_dtype = tile.dtype

            # Deterministic per-record RNG so reruns are reproducible
            rng = random.Random(hash((image_id, row["tile_id"], row["Transform_id"])))

            degraded = apply_stochastic_transform(
                tile,
                order      = str(row["Order"]),
                psf_sigma  = float(row["PSF"]),
                snr_db     = float(row.get("SNR_dB", row.get("SNR (dB)"))), # Fallback handles old/new CSV schema
                source_gsd = float(row["GSD_input"]),
                target_gsd = float(row["GSD_output"]),
                blur_type  = str(row["Blur_Type"]),
                noise_type = str(row["Noise_Type"]),
                rng        = rng,
            )

            # Update geotransform to reflect new pixel size after downsampling
            scale = float(row["GSD_output"]) / float(row["GSD_input"])
            new_transform = Affine(
                transform.a * scale, transform.b, transform.c,
                transform.d, transform.e * scale, transform.f,
            )

            _write_tile(out_path, degraded, new_transform, crs, orig_dtype)
            written += 1

        except Exception:
            logger.exception("Failed: %s  tile=%s  mix=%s", image_id, row["tile_id"], row["Transform_id"])

    return written


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(csv_path: str, source_dir: str, output_dir: str, workers: int = 4) -> None:
    """
    Full pipeline: read CSV → dispatch per-image workers → report.
    """
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    src  = Path(source_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = list(df.groupby("image_id"))
    logger.info("Processing %d image(s), %d total records.", len(groups), len(df))

    total_written = 0

    if workers <= 1:
        for image_id, group in groups:
            n = _process_image(image_id, group, src, out)
            total_written += n
            logger.info("  %-60s  %d/%d tiles done.", image_id, n, len(group))
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for image_id, group in groups:
                f = pool.submit(_process_image, image_id, group, src, out)
                futures[f] = (image_id, len(group))

            for f in as_completed(futures):
                image_id, n_rows = futures[f]
                try:
                    n = f.result()
                    total_written += n
                    total_written_str = f"  %-60s  %d/%d tiles done."
                    logger.info(total_written_str, image_id, n, n_rows)
                except Exception:
                    logger.exception("Worker crashed for %s", image_id)

    logger.info("Finished. %d/%d tiles written to %s", total_written, len(df), out)


# ---------------------------------------------------------------------------
# Entry Point Configuration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Hardcoded Configuration
    CSV_FILE = "tile_mixture_assignments.csv"
    SOURCE_DIRECTORY = "data/raw"
    OUTPUT_DIRECTORY = "data/stochastic"
    WORKERS = 4

    run(
        csv_path   = CSV_FILE,
        source_dir = SOURCE_DIRECTORY,
        output_dir = OUTPUT_DIRECTORY,
        workers    = WORKERS,
    )