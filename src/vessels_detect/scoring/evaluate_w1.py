"""
evaluate_w1.py
--------------
Evaluates the Wasserstein Distance (W1) between two perfectly aligned GeoTIFFs 
(e.g., Synthetic Pléiades vs Real Pléiades or Degraded Néo) by tiling them 
and comparing the pixel intensity distributions of each corresponding tile pair.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from scipy.stats import wasserstein_distance

# Import the existing tiling function from your pipeline
from src.vessels_detect.preprocessing.steps.tiling import tile_image_raw
from src.vessels_detect.scoring.generates_overlap import make_overlap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_tile_w1(tile1_path: Path, tile2_path: Path) -> List[float]:
    """
    Computes the Wasserstein-1 (Earth Mover's) distance between two tiles.
    Calculates the distance per band.
    
    Args:
        tile1_path: Path to the first tile.
        tile2_path: Path to the second tile.
        
    Returns:
        A list of W1 scores, one for each band.
    """
    with rasterio.open(tile1_path) as src1, rasterio.open(tile2_path) as src2:
        # Read all bands into numpy arrays. Shape: (Channels, Height, Width)
        arr1 = src1.read()
        arr2 = src2.read()
        
    if arr1.shape != arr2.shape:
        logger.warning(f"Shape mismatch for tiles {tile1_path.name} and {tile2_path.name}")
        return []

    w1_scores = []
    # Iterate over each band
    for band_idx in range(arr1.shape[0]):
        # Flatten the 2D band arrays into 1D arrays to compare distributions
        band1_flat = arr1[band_idx].flatten()
        band2_flat = arr2[band_idx].flatten()
        
        # Calculate W1 distance
        w1 = wasserstein_distance(band1_flat, band2_flat)
        w1_scores.append(w1)
        
    return w1_scores


def evaluate_images(
    tif_path_1: Path, 
    tif_path_2: Path, 
    out_dir: Path, 
    tile_size: int = 1024, 
    overlap: int = 0
) -> None:
    """
    Tiles two images, matches corresponding tiles, computes W1, and prints stats.
    """
    logger.info(f"Evaluating {tif_path_1.name} vs {tif_path_2.name}")
    
    # 1. Create output directories for the temporary tiles
    dir1 = out_dir / "image1_tiles"
    dir2 = out_dir / "image2_tiles"
    dir1.mkdir(parents=True, exist_ok=True)
    dir2.mkdir(parents=True, exist_ok=True)
    
    # 2. Tile both images using the pipeline's raw tiling function
    # Returns List[Tuple[Path, x_off, y_off]]
    logger.info("Tiling Image 1...")
    tiles_1 = tile_image_raw(tif_path_1, dir1, tile_size, overlap, compress="lzw")
    
    logger.info("Tiling Image 2...")
    tiles_2 = tile_image_raw(tif_path_2, dir2, tile_size, overlap, compress="lzw")
    
    # 3. Create dictionaries mapping (x_off, y_off) to the tile path
    # This guarantees we only compare perfectly overlapping spatial regions
    map1: Dict[Tuple[int, int], Path] = {(x, y): path for path, x, y in tiles_1}
    map2: Dict[Tuple[int, int], Path] = {(x, y): path for path, x, y in tiles_2}
    
    # Find common tile coordinates (in case one image had uniform/blank tiles that were skipped)
    common_coords = set(map1.keys()).intersection(set(map2.keys()))
    
    if not common_coords:
        logger.error("No matching tiles found between the two images.")
        return
        
    logger.info(f"Found {len(common_coords)} perfectly matching tile pairs.")
    
    # 4. Compute W1 for each pair
    # We will store the average W1 score across all bands for each tile
    mean_w1_per_tile = []
    
    for (x, y) in sorted(common_coords):
        t1_path = map1[(x, y)]
        t2_path = map2[(x, y)]
        
        # Get W1 scores per band
        w1_bands = compute_tile_w1(t1_path, t2_path)
        
        if w1_bands:
            # Average W1 across all spectral bands for this specific tile
            avg_w1 = np.mean(w1_bands)
            mean_w1_per_tile.append(avg_w1)
            logger.debug(f"Tile at {x},{y} | Mean W1: {avg_w1:.4f} | Per band: {w1_bands}")

    if not mean_w1_per_tile:
        logger.error("Could not compute W1 scores (perhaps due to shape mismatches).")
        return

    # 5. Compute global statistics
    mean_w1_per_tile = np.array(mean_w1_per_tile)
    
    mean_val = np.mean(mean_w1_per_tile)
    median_val = np.median(mean_w1_per_tile)
    std_val = np.std(mean_w1_per_tile)
    min_val = np.min(mean_w1_per_tile)
    max_val = np.max(mean_w1_per_tile)
    
    # 6. Print the Detailed Report
    print("\n" + "="*50)
    print(" 📊 WASSERSTEIN DISTANCE (W1) REPORT")
    print("="*50)
    print(f"Tiles Compared : {len(mean_w1_per_tile)}")
    print(f"Tile Size      : {tile_size}x{tile_size} px")
    print(f"Image 1        : {tif_path_1.name}")
    print(f"Image 2        : {tif_path_2.name}")
    print("-" * 50)
    print(f"Mean W1        : {mean_val:.4f}  <-- Overall radiometric distance")
    print(f"Median W1      : {median_val:.4f}  <-- Robust to outlier tiles")
    print(f"Std Deviation  : {std_val:.4f}  <-- Consistency across the image")
    print(f"Min W1         : {min_val:.4f}  <-- Best matching tile")
    print(f"Max W1         : {max_val:.4f}  <-- Worst matching tile")
    print("="*50 + "\n")


if __name__ == "__main__":
    image1_raw = Path("data/scoring/merged_degraded.tif") # synthetic path
    image2_raw = Path("data/scoring/merged.tif") # real path
    
    base_out_dir = Path("./w1_evaluation_tmp")
    
    # 1. Create the perfect intersection first
    logger.info("Aligning image footprints...")
    image1_cropped, image2_cropped = make_overlap(
        tif_path_1=image1_raw, 
        tif_path_2=image2_raw, 
        out_dir=base_out_dir / "aligned_inputs"
    )
    
    # 2. Run the evaluation on the perfectly aligned images
    evaluate_images(
        tif_path_1=image1_cropped, 
        tif_path_2=image2_cropped, 
        out_dir=base_out_dir / "tiles", 
        tile_size=1024, 
        overlap=0
    )