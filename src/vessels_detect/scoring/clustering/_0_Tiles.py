import math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

def _pad_tile(arr: np.ndarray, tile_size: int, fill_value: float = 0.0) -> np.ndarray:
    """Zero-pad (or nodata-pad) (C, H, W) to (C, tile_size, tile_size)."""
    pad_h = tile_size - arr.shape[1]
    pad_w = tile_size - arr.shape[2]
    
    if pad_h == 0 and pad_w == 0:
        return arr
        
    return np.pad(
        arr,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=fill_value,
    )

def _build_tile_profile(src: rasterio.DatasetReader, tile_transform, tile_size: int, compress: str = "lzw") -> dict:
    """Construct a rasterio write profile preserving source dtype and CRS."""
    return {
        "driver":     "GTiff",
        "dtype":      src.dtypes[0],       
        "count":      src.count,            
        "width":      tile_size,
        "height":     tile_size,
        "crs":        src.crs,
        "transform":  tile_transform,
        "compress":   compress,
        "predictor":  2,                    
        "tiled":      True,
        "blockxsize": min(256, tile_size),
        "blockysize": min(256, tile_size),
    }

def process_single_tif(tif_path: Path, output_image_dir: Path, tile_size: int = 1024):
    """Tiles a single GeoTIFF into fixed-size patches with 0 overlap."""
    stem = tif_path.stem

    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        nodata = src.nodata
        fill_value = float(nodata) if nodata is not None else 0.0

        # With 0 overlap, stride is exactly the tile_size
        stride = tile_size
        n_cols = math.ceil(W / stride)
        n_rows = math.ceil(H / stride)

        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                x_off = col_idx * stride
                y_off = row_idx * stride

                # Actual window extent clipped to image bounds
                win_w = min(tile_size, W - x_off)
                win_h = min(tile_size, H - y_off)

                window = Window(x_off, y_off, win_w, win_h)
                raw_tile = src.read(window=window)

                # Pad edge tiles to uniform tile_size x tile_size
                padded = _pad_tile(raw_tile, tile_size, fill_value)

                # Discard tiles where every pixel is identical (uniform blank/nodata scene)
                if padded.min() == padded.max():
                    continue

                tile_transform = src.window_transform(window)
                profile = _build_tile_profile(src, tile_transform, tile_size)

                # Save the tile
                out_path = output_image_dir / f"{stem}_{x_off}_{y_off}.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(padded)
                    # Embed provenance tags just like the original tiling.py
                    dst.update_tags(
                        source_tif=tif_path.name,
                        col_off=str(x_off),
                        row_off=str(y_off),
                        src_width=str(W),
                        src_height=str(H),
                        tile_size=str(tile_size),
                    )

if __name__ == "__main__":
    # 1. Define your 3 input folders here
    input_folders = [
        Path("data/scoring/diff/Gen"),
        Path("data/scoring/diff/PL"),
        Path("data/scoring/diff/PLN")
    ]
    
    # 2. Define the main 4th folder where outputs will go
    output_base_dir = Path("data/scoring/tiled")
    
    TILE_SIZE = 320

    print(f"Starting tiling process. Output directory: {output_base_dir}")

    for input_folder in input_folders:
        if not input_folder.exists():
            print(f"Warning: Input folder {input_folder} does not exist. Skipping.")
            continue
            
        # Create corresponding subfolder inside the main output directory
        subfolder_name = input_folder.name
        output_sub_dir = output_base_dir / subfolder_name
        output_sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Grab all tifs in the current input folder
        tif_files = list(input_folder.glob("*.tif"))
        
        if not tif_files:
            print(f"No .tif files found in {input_folder}")
            continue
            
        print(f"\nProcessing '{subfolder_name}' ({len(tif_files)} images)...")
        
        for tif_path in tqdm(tif_files, desc=f"Tiling {subfolder_name}", unit="img"):
            try:
                process_single_tif(
                    tif_path=tif_path, 
                    output_image_dir=output_sub_dir, 
                    tile_size=TILE_SIZE
                )
            except Exception as e:
                print(f"Error processing {tif_path.name}: {e}")

    print("\nDone! All tiles generated successfully.")