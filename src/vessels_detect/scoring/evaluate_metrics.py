import logging
import random
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from tqdm import tqdm

try:
    import piq
except ImportError:
    raise SystemExit("ERROR: 'piq' is required. Run: pip install piq")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BRISQUE_DN_MAX = 65535.0
MAX_PAIRS_PER_COMPARE = 200  
BIT_DEPTH = 16
NODATA = 0

# ---------------------------------------------------------------------------
# Metric Calculation Functions
# ---------------------------------------------------------------------------

def _compute_histogram(arr: np.ndarray, max_val: int, nodata_val: int | None) -> np.ndarray:
    if nodata_val is not None:
        arr = arr[arr != nodata_val]
    return np.bincount(arr.flatten(), minlength=max_val)

def _compute_radial_profile(image: np.ndarray) -> np.ndarray:
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)**2
    
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.indices((h, w))
    radii = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(np.int32)
    
    tbin = np.bincount(radii.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(radii.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    return 10 * np.log10(radial_profile + 1e-10)

def compute_image_metrics(tif_path: Path | str) -> dict:
    metrics = {}
    
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        arr_float = arr.astype(np.float32)

    # 1. Histogram (for W1)
    max_val = 256 if BIT_DEPTH == 8 else 65536
    metrics['histogram'] = _compute_histogram(arr, max_val, NODATA)
    
    # 2. FFT Radial Profile
    if NODATA is not None:
        mask = arr_float == NODATA
        if np.any(mask):
            arr_float[mask] = np.mean(arr_float[~mask]) if not np.all(mask) else 0.0
    metrics['fft_profile'] = _compute_radial_profile(arr_float)

    # 3. Piq Metrics (BRISQUE & TV)
    t_arr = arr.astype(np.float32)
    if NODATA is not None:
        t_arr[t_arr == NODATA] = 0.0
    t_arr /= _BRISQUE_DN_MAX
    t_arr = np.clip(t_arr, 0.0, 1.0)
    
    tensor = torch.from_numpy(t_arr).unsqueeze(0).unsqueeze(0).to(_DEVICE)
    
    with torch.no_grad():
        metrics['brisque'] = float(piq.brisque(tensor, data_range=1.0, reduction="mean"))
        metrics['tv'] = float(piq.total_variation(tensor, reduction="mean", norm_type="l2"))
        
    return metrics

def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, cmap: str, output_path: Path):
    """Generates and saves a seaborn heatmap from a pairwise dataframe."""
    pivot_df = df.pivot(index='Source Cluster', columns='Target Cluster', values=value_col)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap=cmap, cbar=True, square=True, 
                linewidths=0.5, linecolor='gray')
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = Path("data/scoring/tiled")
    FOLDERS = ["Gen", "PL", "PLN"]
    
    # Create an output directory for CSVs and PNGs
    OUTPUT_DIR = BASE_DIR / "evaluation_outputs"
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Load Clusters
    cluster_dict = {}
    print("📂 Loading cluster distributions...")
    
    for g in FOLDERS:
        csv_path = BASE_DIR / g / "clustering_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for c in sorted(df['cluster'].unique()):
                key = f"{g} - Cluster {c}"
                cluster_dict[key] = df[df['cluster'] == c]['chip_path'].tolist()
                
    if not cluster_dict:
        raise SystemExit("No clustering_results.csv found. Run Step 2 first.")

    # 2. Pre-compute metrics per image
    print("\n⚙️  Pre-computing image features (TV, BRISQUE, FFT, Histograms)...")
    image_cache = {}
    
    for cluster_name, paths in cluster_dict.items():
        for p in tqdm(paths, desc=f"  {cluster_name}", leave=False):
            if p not in image_cache:
                try:
                    image_cache[p] = compute_image_metrics(p)
                except Exception as e:
                    logger.warning(f"Failed to process {p}: {e}")

    # 3. Compute Per-Cluster Single Metrics
    print("\n" + "═"*70)
    print("  📐 PER-CLUSTER AVERAGES (Total Variation & BRISQUE)")
    print("═"*70)
    
    cluster_metrics_data = []
    
    for cluster_name, paths in cluster_dict.items():
        valid_paths = [p for p in paths if p in image_cache]
        if not valid_paths:
            continue
            
        mean_tv = np.mean([image_cache[p]['tv'] for p in valid_paths])
        mean_bq = np.mean([image_cache[p]['brisque'] for p in valid_paths])
        
        cluster_metrics_data.append({
            "Cluster": cluster_name,
            "Mean TV": mean_tv,
            "Mean BRISQUE": mean_bq
        })
        print(f"  {cluster_name:<20} | TV: {mean_tv:<10.6f} | BQ: {mean_bq:<10.2f}")

    # Save Cluster Metrics CSV
    df_cluster = pd.DataFrame(cluster_metrics_data)
    df_cluster.to_csv(OUTPUT_DIR / "cluster_metrics.csv", index=False)

    # 4. Compute Pairwise Cluster Metrics (W1 & FFT ΔdB)
    print("\n" + "═"*90)
    print("  🔬 PAIRWISE QUALITY GRID (Mean W1 & Mean FFT ΔdB)")
    print("═"*90)

    cluster_keys = list(cluster_dict.keys())
    bins = np.arange(256 if BIT_DEPTH == 8 else 65536)
    
    pairwise_data = []

    for src_key, tgt_key in combinations_with_replacement(cluster_keys, 2):
        src_paths = [p for p in cluster_dict[src_key] if p in image_cache]
        tgt_paths = [p for p in cluster_dict[tgt_key] if p in image_cache]
        
        if not src_paths or not tgt_paths:
            continue

        num_pairs = min(MAX_PAIRS_PER_COMPARE, len(src_paths) * len(tgt_paths))
        sampled_src = random.choices(src_paths, k=num_pairs)
        sampled_tgt = random.choices(tgt_paths, k=num_pairs)
        
        w1_scores = []
        fft_scores = []
        
        for p_s, p_t in zip(sampled_src, sampled_tgt):
            data_s = image_cache[p_s]
            data_t = image_cache[p_t]
            
            w1 = wasserstein_distance(bins, bins, data_s['histogram'], data_t['histogram'])
            w1_scores.append(w1)
            
            prof_s = data_s['fft_profile']
            prof_t = data_t['fft_profile']
            min_len = min(len(prof_s), len(prof_t))
            mae = np.mean(np.abs(prof_s[:min_len] - prof_t[:min_len]))
            fft_scores.append(mae)

        mean_w1 = np.mean(w1_scores)
        mean_fft = np.mean(fft_scores)
        
        print(f"  {src_key:<20} | {tgt_key:<20} | W1: {mean_w1:<10.2f} | FFT: {mean_fft:<10.4f}")
        
        # Append to our dataset. If src and tgt are different, mirror the data for a full matrix
        pairwise_data.append({"Source Cluster": src_key, "Target Cluster": tgt_key, "Mean W1": mean_w1, "Mean FFT ΔdB": mean_fft})
        if src_key != tgt_key:
            pairwise_data.append({"Source Cluster": tgt_key, "Target Cluster": src_key, "Mean W1": mean_w1, "Mean FFT ΔdB": mean_fft})

    print("═"*90)

    # Save Pairwise Metrics CSV
    df_pairwise = pd.DataFrame(pairwise_data)
    df_pairwise.to_csv(OUTPUT_DIR / "pairwise_metrics.csv", index=False)

    # 5. Generate Heatmap PNGs
    print(f"\n📊 Generating Heatmap PNG grids in {OUTPUT_DIR}...")
    
    # For W1 and FFT, lower is better. "YlGnBu" (Yellow-Green-Blue) is a good colormap 
    # where lower values are lighter and higher values are darker.
    plot_heatmap(
        df=df_pairwise, 
        value_col="Mean W1", 
        title="Wasserstein-1 Distance (Lower = More Similar Histograms)", 
        cmap="YlGnBu", 
        output_path=OUTPUT_DIR / "grid_W1.png"
    )
    
    plot_heatmap(
        df=df_pairwise, 
        value_col="Mean FFT ΔdB", 
        title="FFT Power Spectral Density Error (Lower = More Similar Sharpness)", 
        cmap="YlOrRd", 
        output_path=OUTPUT_DIR / "grid_FFT.png"
    )

    print("✅ All CSVs and PNG grids successfully saved!")