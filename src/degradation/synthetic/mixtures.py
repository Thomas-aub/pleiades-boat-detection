import random
import pandas as pd
import json
import logging
from pathlib import Path
import rasterio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
log = logging.getLogger(__name__)


def compute_tiling_native_step(input_directory: str, tile_size: int):
    """
    Logic:
    1. Reads TIF headers (Native CRS).
    2. Computes the distance in meters for 'tile_size' pixels (step = tile_size * GSD).
    3. Calculates corner coordinates by applying this meter-step from the origin.
    4. Outputs borders and corners in the original coordinate system.
    """
    input_dir = Path(input_directory)
    tif_files = sorted(input_dir.glob("*.tif"))
    
    all_records = []
    dots_features = []
    lines_features = []

    for tif_path in tif_files:
        log.info(f"Processing: {tif_path.name}")
        
        try:
            with rasterio.open(tif_path) as ds:
                if ds.crs is None:
                    log.warning(f"Skipping {tif_path.name}: Missing CRS.")
                    continue
                
                # Image metadata
                transform = ds.transform
                width_px = ds.width
                height_px = ds.height
                
                # Native GSD (Resolution)
                gsd_x = abs(transform.a)
                gsd_y = abs(transform.e)
                
                # Compute length in meters for the tile_size
                step_x_meters = tile_size * gsd_x
                step_y_meters = tile_size * gsd_y
                
                # Iterate through the image pixels to compute native CRS coordinates
                for row_idx in range(0, height_px, tile_size):
                    for col_idx in range(0, width_px, tile_size):
                        
                        # Determine current tile pixel boundaries (handling edge clipping)
                        c0, r0 = col_idx, row_idx
                        c1 = min(col_idx + tile_size, width_px)
                        r1 = min(row_idx + tile_size, height_px)

                        # Transform pixel indices to Native CRS coordinates
                        # top-left corner
                        left, top = transform * (c0, r0)
                        # bottom-right corner
                        right, bottom = transform * (c1, r1)
                        
                        # Set corners (TL, TR, BR, BL)
                        xs = [left, right, right, left]
                        ys = [top, top, bottom, bottom]
                        
                        tile_id = f"{row_idx}_{col_idx}"
                        
                        # --- 1. CSV RECORD ---
                        all_records.append({
                            "image_id": tif_path.name,
                            "tile_id": tile_id,
                            "min_x": min(xs),
                            "max_x": max(xs),
                            "min_y": min(ys),
                            "max_y": max(ys),
                            "tile_size_px": tile_size,
                            "length_x_meters": step_x_meters,
                            "length_y_meters": step_y_meters,
                            "native_crs": ds.crs.to_string()
                        })

                        # --- 2. GEOJSON DOTS (Corners) ---
                        labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
                        for i in range(4):
                            dots_features.append({
                                "type": "Feature",
                                "properties": {
                                    "image_id": tif_path.name, 
                                    "tile_id": tile_id, 
                                    "corner": labels[i]
                                },
                                "geometry": {"type": "Point", "coordinates": [xs[i], ys[i]]}
                            })

                        # --- 3. GEOJSON BORDERS (Lines) ---
                        lines_features.append({
                            "type": "Feature",
                            "properties": {"image_id": tif_path.name, "tile_id": tile_id},
                            "geometry": {
                                "type": "LineString", 
                                "coordinates": [
                                    [xs[0], ys[0]], [xs[1], ys[1]], 
                                    [xs[2], ys[2]], [xs[3], ys[3]], [xs[0], ys[0]]
                                ]
                            }
                        })
                        
        except Exception as e:
            log.error(f"Error processing {tif_path.name}: {e}")

    # Export results
    if all_records:
        pd.DataFrame(all_records).to_csv("tile_bboxes_native.csv", index=False)
        
        with open("tile_corners_native.geojson", "w") as f:
            json.dump({"type": "FeatureCollection", "features": dots_features}, f)
            
        with open("tile_borders_native.geojson", "w") as f:
            json.dump({"type": "FeatureCollection", "features": lines_features}, f)
        
        log.info(f"Done! Created tile bboxes based on {tile_size}px ({step_x_meters:.2f}m) steps.")
    else:
        log.warning("No data produced.")



def generate_image_degradation_mixtures(psf_range, snr_range, num_mixtures):
    """Generates the global pool of randomized degradation mixtures."""
    mixtures = []
    steps = ["B", "D", "N"]
    
    for i in range(num_mixtures):
        current_order = steps.copy()
        random.shuffle(current_order)
        
        selected_snr = random.uniform(snr_range[0], snr_range[1])
        selected_psf = random.uniform(psf_range[0], psf_range[1])
        
        mixture = {
            "Transform_id": i,
            "step_order": current_order,
            "snr_db": round(selected_snr, 2),
            "psf_size": round(selected_psf, 2)
        }
        mixtures.append(mixture)
        
    return mixtures

def assign_mixtures_to_tiles(
    psf_limits: list, 
    snr_limits: list, 
    n_mixtures: int, 
    n_mixtures_per_tiles: int, 
    input_directory: str, 
    GSD_output: float, 
    tile_size: int = 1024
):
    """
    Reads TIFs, computes native tiling and input GSD, 
    and randomly assigns mixtures to each tile.
    Exports the result as a CSV.
    """
    # Safety check: Ensure we have enough mixtures in the pool to sample uniquely per tile
    if n_mixtures_per_tiles > n_mixtures:
        raise ValueError("n_mixtures_per_tiles cannot be greater than the total n_mixtures pool.")

    # 1. Generate the global pool of mixtures
    log.info(f"Generating global pool of {n_mixtures} mixtures...")
    mixtures_pool = generate_image_degradation_mixtures(psf_limits, snr_limits, n_mixtures)
    
    input_dir = Path(input_directory)
    tif_files = sorted(input_dir.glob("*.tif"))
    
    all_records = []

    # 2. Iterate over the TIF files to compute tiles
    for tif_path in tif_files:
        log.info(f"Processing: {tif_path.name}")
        
        try:
            with rasterio.open(tif_path) as ds:
                if ds.crs is None:
                    log.warning(f"Skipping {tif_path.name}: Missing CRS.")
                    continue
                
                transform = ds.transform
                width_px = ds.width
                height_px = ds.height
                
                # Compute Input GSD (mean of pixel width and height)
                gsd_x = abs(transform.a)
                gsd_y = abs(transform.e)
                gsd_input = (gsd_x + gsd_y) / 2.0
                
                # Physical step sizes
                step_x_meters = tile_size * gsd_x
                step_y_meters = tile_size * gsd_y
                
                for row_idx in range(0, height_px, tile_size):
                    for col_idx in range(0, width_px, tile_size):
                        
                        # Tile boundaries in pixel space
                        c0, r0 = col_idx, row_idx
                        c1 = min(col_idx + tile_size, width_px)
                        r1 = min(row_idx + tile_size, height_px)

                        # Convert to Native CRS coordinates
                        left, top = transform * (c0, r0)
                        right, bottom = transform * (c1, r1)
                        
                        xs = [left, right, right, left]
                        ys = [top, top, bottom, bottom]
                        
                        tile_id = f"{row_idx}_{col_idx}"
                        
                        # 3. Independently select mixtures for THIS specific tile
                        # Using random.sample ensures the same tile gets unique transforms from the pool
                        selected_mixtures = random.sample(mixtures_pool, k=n_mixtures_per_tiles)
                        
                        # 4. Create a record for every selected mixture on this tile
                        for mix in selected_mixtures:
                            all_records.append({
                                "image_id": tif_path.name,
                                "tile_id": tile_id,
                                "min_x": min(xs),
                                "max_x": max(xs),
                                "min_y": min(ys),
                                "max_y": max(ys),
                                "tile_size_px": tile_size,
                                "length_x_meters": step_x_meters,
                                "length_y_meters": step_y_meters,
                                "native_crs": ds.crs.to_string(),
                                "GSD_input": round(gsd_input, 4),
                                "GSD_output": GSD_output,
                                "Transform_id": mix["Transform_id"],
                                "Order": " -> ".join(mix["step_order"]),
                                "SNR (dB)": mix["snr_db"],
                                "PSF": mix["psf_size"]
                            })
                            
        except Exception as e:
            log.error(f"Error processing {tif_path.name}: {e}")

    # 5. Export to CSV
    if all_records:
        df = pd.DataFrame(all_records)
        output_filename = "tile_mixture_assignments.csv"
        df.to_csv(output_filename, index=False)
        log.info(f"Successfully generated {len(df)} rows and saved to {output_filename}")
        return df
    else:
        log.warning("No data was produced.")
        return None

if __name__ == "__main__":
    # Example execution:
    df_result = assign_mixtures_to_tiles(
        psf_limits=[0.8, 2.5],
        snr_limits=[15, 35],
        n_mixtures=10,                  # Global pool of 50 unique mixtures
        n_mixtures_per_tiles=3,         # Randomly pick 3 from the 50 for EVERY tile
        input_directory="/home/thomas/Documents/code/pleiades-boat-detection/data/raw",
        GSD_output=0.5,
        tile_size=1024
    )


