"""Global image quality evaluation: Wasserstein W1, BRISQUE, Total Variation, and FFT.

Metrics
-------
W1 (Wasserstein-1)
    Per-pair histogram transport distance. Scene-content confounds cross-location
    comparisons; only same-scene pairs carry emulation signal.

FFT ΔdB (Power Spectral Density Mean Absolute Error)
    Measures the difference in spatial frequency energy (sharpness/MTF) between two 
    images. A native-resolution center crop is used to avoid aliasing. Lower = tighter
    match in blur and texture characteristics.

BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    No-reference IQA in the spatial domain (piq.brisque). Lower = fewer natural
    scene statistics distortions. Computed per-pair on band 1.

Total Variation
    Per-image L2 TV norm (piq.total_variation). Measures spatial roughness /
    sharpness. Higher = more high-frequency content. Computed once per image.

Dependencies
------------
    pip install piq rasterio scipy numpy torch tqdm
"""

import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import rasterio
import torch
from scipy.stats import wasserstein_distance

try:
    import piq
except ImportError:
    raise SystemExit("ERROR: 'piq' is required. Run: pip install piq")

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("ERROR: 'tqdm' is required. Run: pip install tqdm")


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BRISQUE expects float32 in [0, 1]. We normalise 16-bit imagery to this range.
_BRISQUE_DN_MAX = 65535.0

# Maximum pixel rows to sample when computing BRISQUE / TV on large images.
_MAX_BRISQUE_SIDE = 2048

# Size of the center crop used for FFT calculation (must be native resolution)
_FFT_CROP_SIZE = 2048


# ---------------------------------------------------------------------------
# Histogram / W1
# ---------------------------------------------------------------------------

def _compute_histogram(tif_path: Path, max_val: int, nodata_val: int | None) -> np.ndarray:
    """Build a full-image histogram band-by-band using block reads."""
    hist = np.zeros(max_val, dtype=np.int64)
    with rasterio.open(tif_path) as src:
        for _, window in src.block_windows(1):
            block = src.read(1, window=window)
            if nodata_val is not None:
                block = block[block != nodata_val]
            hist += np.bincount(block.flatten(), minlength=max_val)
    return hist


def compute_w1(
    path_a: Path,
    path_b: Path,
    bit_depth: int = 16,
    nodata_val: int | None = None,
) -> float:
    """Wasserstein-1 distance between the band-1 histograms of two images."""
    max_val = 256 if bit_depth == 8 else 65536
    hist_a = _compute_histogram(path_a, max_val, nodata_val)
    hist_b = _compute_histogram(path_b, max_val, nodata_val)
    bins   = np.arange(max_val)
    return float(wasserstein_distance(bins, bins, hist_a, hist_b))


# ---------------------------------------------------------------------------
# FFT / Power Spectral Density (PSD)
# ---------------------------------------------------------------------------

def _compute_radial_profile(image: np.ndarray) -> np.ndarray:
    """Computes the 1D radially averaged power spectrum of a 2D image."""
    # Compute 2D Fast Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    
    # Calculate Power (Magnitude squared)
    magnitude_spectrum = np.abs(f_shift)**2
    
    # Calculate radial distance from center for each pixel
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.indices((h, w))
    radii = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(np.int32)
    
    # Average the energy across each radial bin (frequency band)
    tbin = np.bincount(radii.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(radii.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    
    # Return logarithmic scale (dB)
    return 10 * np.log10(radial_profile + 1e-10)


def compute_fft_distance(
    path_a: Path, 
    path_b: Path, 
    nodata_val: int | None = None, 
    crop_size: int = _FFT_CROP_SIZE
) -> float:
    """
    Mean Absolute Error between the 1D Power Spectral Density of two images.
    Uses a native-resolution center crop to preserve true high-frequency content.
    """
    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        h_a, w_a = src_a.height, src_a.width
        h_b, w_b = src_b.height, src_b.width

        # Ensure we extract the exact same pixel dimensions for fair FFT comparison
        h = min(h_a, h_b, crop_size)
        w = min(w_a, w_b, crop_size)

        off_y_a, off_x_a = (h_a - h) // 2, (w_a - w) // 2
        off_y_b, off_x_b = (h_b - h) // 2, (w_b - w) // 2

        win_a = rasterio.windows.Window(off_x_a, off_y_a, w, h)
        win_b = rasterio.windows.Window(off_x_b, off_y_b, w, h)

        arr_a = src_a.read(1, window=win_a).astype(np.float32)
        arr_b = src_b.read(1, window=win_b).astype(np.float32)

    if nodata_val is not None:
        # Fill NoData with the mean of the crop to avoid artificial sharp edges 
        # that would heavily pollute the high-frequency spectrum.
        mask_a = arr_a == nodata_val
        if np.any(mask_a):
            arr_a[mask_a] = np.mean(arr_a[~mask_a]) if not np.all(mask_a) else 0.0
            
        mask_b = arr_b == nodata_val
        if np.any(mask_b):
            arr_b[mask_b] = np.mean(arr_b[~mask_b]) if not np.all(mask_b) else 0.0

    prof_a = _compute_radial_profile(arr_a)
    prof_b = _compute_radial_profile(arr_b)

    # MAE (Mean Absolute Error) between the two curves in dB
    min_len = min(len(prof_a), len(prof_b))
    mae = np.mean(np.abs(prof_a[:min_len] - prof_b[:min_len]))
    
    return float(mae)


# ---------------------------------------------------------------------------
# piq helpers — load band as (1, 1, H, W) float32 tensor in [0, 1]
# ---------------------------------------------------------------------------

def _load_tensor(
    tif_path: Path,
    nodata_val: int | None,
    max_side: int = _MAX_BRISQUE_SIDE,
) -> torch.Tensor:
    """Read band 1, sub-sample if necessary, mask nodata, return 4-D tensor."""
    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width

        # Compute sub-sample step so the longest axis <= max_side
        step = max(1, max(h, w) // max_side)
        row_indices = list(range(0, h, step))
        col_indices = list(range(0, w, step))

        band = src.read(1).astype(np.float32)

    band = band[np.ix_(row_indices, col_indices)]

    if nodata_val is not None:
        band[band == nodata_val] = 0.0

    band /= _BRISQUE_DN_MAX
    band  = np.clip(band, 0.0, 1.0)

    return torch.from_numpy(band).unsqueeze(0).unsqueeze(0).to(_DEVICE)


# ---------------------------------------------------------------------------
# Per-pair BRISQUE
# ---------------------------------------------------------------------------

def compute_brisque(
    path_a: Path,
    path_b: Path,
    nodata_val: int | None = None,
) -> tuple[float, float]:
    """Return (brisque_a, brisque_b) for the two images."""
    ta = _load_tensor(path_a, nodata_val)
    tb = _load_tensor(path_b, nodata_val)
    with torch.no_grad():
        sa = float(piq.brisque(ta, data_range=1.0, reduction="mean"))
        sb = float(piq.brisque(tb, data_range=1.0, reduction="mean"))
    return sa, sb


# ---------------------------------------------------------------------------
# Per-image Total Variation
# ---------------------------------------------------------------------------

def compute_tv(tif_path: Path, nodata_val: int | None = None) -> float:
    """L2 Total Variation of band 1, mean-reduced over pixels."""
    t = _load_tensor(tif_path, nodata_val)
    with torch.no_grad():
        return float(piq.total_variation(t, reduction="mean", norm_type="l2"))


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------

_NAME_MAP: dict[str, str] = {
    "PLN_2_PL_center": "Neo→PHR center",
    "PLN_2_PL_bottom": "Neo→PHR bottom",
    "PLN_center":      "Neo center",
    "PLN_bottom":      "Neo bottom",
    "PL_real_top":     "PHR top",
}

def _rename(filename: str) -> str:
    name = filename.replace(".tif", "")
    for key, label in _NAME_MAP.items():
        if key in name:
            return name.replace(key, label)
    return name


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _hbar(n: int, char: str = "─") -> str:
    return char * n

def _print_report(
    pair_results: list[tuple[str, str, float, float, float, float]],
    tv_results:   dict[str, float],
    col_w:        int,
) -> None:
    W1_W  = 10
    FFT_W = 10
    BQ_W  = 12

    # ── Section 1: pairwise ────────────────────────────────────────────────
    hdr = (
        f"  {'Source':<{col_w}}   {'Target':<{col_w}}"
        f"   {'W1':>{W1_W}}   {'FFT ΔdB':>{FFT_W}}   {'BRISQUE src':>{BQ_W}}   {'BRISQUE tgt':>{BQ_W}}   Δ"
    )
    bar = _hbar(len(hdr), "═")
    sep = _hbar(len(hdr), "─")

    print()
    print(bar)
    print("  🔬  PAIRWISE QUALITY  —  W1 · FFT · BRISQUE")
    print(bar)
    print(hdr)
    print(sep)

    for src, tgt, w1, fft_dist, bq_s, bq_t in pair_results:
        delta  = bq_t - bq_s
        symbol = "▲ worse" if delta > 0.5 else ("▼ better" if delta < -0.5 else "~ equal")
        print(
            f"  {src:<{col_w}}   {tgt:<{col_w}}"
            f"   {w1:>{W1_W}.4f}   {fft_dist:>{FFT_W}.4f}   {bq_s:>{BQ_W}.2f}   {bq_t:>{BQ_W}.2f}   {symbol}"
        )

    print(bar)
    print("  W1: lower = identical histogram  |  FFT ΔdB: lower = identical sharpness/MTF")
    print("  BRISQUE: lower = less distorted.  Δ = BRISQUE_tgt − BRISQUE_src")

    # ── Section 2: per-image TV ────────────────────────────────────────────
    TV_W    = 14
    BAR_LEN = 24
    hdr2    = f"  {'Image':<{col_w}}   {'Total Variation':>{TV_W}}   {'':24}"
    bar2    = _hbar(len(hdr2), "═")
    sep2    = _hbar(len(hdr2), "─")

    tv_max = max(tv_results.values()) if tv_results else 1.0

    print()
    print(bar2)
    print("  📐  PER-IMAGE TOTAL VARIATION  —  L2 norm, mean over pixels")
    print(bar2)
    print(hdr2)
    print(sep2)

    for name, tv in sorted(tv_results.items(), key=lambda x: x[1], reverse=True):
        filled = int((tv / tv_max) * BAR_LEN)
        bar_str = "█" * filled + "░" * (BAR_LEN - filled)
        print(f"  {name:<{col_w}}   {tv:>{TV_W}.6f}   {bar_str}")

    print(bar2)
    print("  Higher TV = more spatial energy (sharper / more textured).")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    folder    = Path("data/scoring/diff")
    BIT_DEPTH = 16
    NODATA    = 0

    if not folder.exists():
        raise SystemExit(f"❌ Directory not found: {folder.absolute()}")

    tif_files = sorted(folder.glob("*.tif"))
    if len(tif_files) < 2:
        raise SystemExit(f"⚠️  Need at least 2 .tif files, found {len(tif_files)}.")

    print(f"\n🔎 Scanning : {folder}")
    print(f"📂 Files    : {len(tif_files)}")
    print(f"💻 Device   : {_DEVICE}\n")

    # Per-image TV 
    tv_results: dict[str, float] = {}
    print("Computing Total Variation per image...")
    for p in tqdm(tif_files, desc="  TV", leave=False):
        tv_results[_rename(p.name)] = compute_tv(p, nodata_val=NODATA)

    # Per-pair W1 + FFT + BRISQUE
    pair_results: list[tuple[str, str, float, float, float, float]] = []
    print("Computing W1, FFT, and BRISQUE per pair...")
    for path_a, path_b in tqdm(list(combinations(tif_files, 2)), desc="  pairs", leave=False):
        logger.info(f"  {path_a.name}  vs  {path_b.name}")
        try:
            w1         = compute_w1(path_a, path_b, BIT_DEPTH, NODATA)
            fft_dist   = compute_fft_distance(path_a, path_b, NODATA)
            bq_a, bq_b = compute_brisque(path_a, path_b, NODATA)
            
            pair_results.append(
                (_rename(path_a.name), _rename(path_b.name), w1, fft_dist, bq_a, bq_b)
            )
        except Exception as exc:
            logger.error(f"Failed on {path_a.name} vs {path_b.name}: {exc}")

    if pair_results:
        col_w = max(
            max(len(r[0]) for r in pair_results),
            max(len(r[1]) for r in pair_results),
            max(len(k)    for k in tv_results),
            20,
        )
        _print_report(pair_results, tv_results, col_w)
