import rasterio
from rasterio.enums import Resampling

def downsample_tif_x2(input_path, output_path):
    """
    Downsamples a TIF file by 2x using Bicubic resampling 
    while preserving geospatial metadata.
    """
    with rasterio.open(input_path) as src:
        # Calculate new dimensions (downsample by factor of 2)
        new_height = int(src.height / 2)
        new_width = int(src.width / 2)
        
        # Read the data and perform resampling
        # Resampling.cubic is the equivalent of bicubic
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.cubic
        )

        # Scale the transform (the geospatial matrix) 
        # so the smaller image still points to the correct place on Earth
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )

        # Update the metadata profile for the new file
        profile = src.profile.copy()
        profile.update({
            'transform': transform,
            'width': new_width,
            'height': new_height
        })

        # Write the downsampled file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
            
    print(f"Downsampled: {input_path} -> {output_path}")

# --- Simple Main Loop to process your PL folder ---
import os
from glob import glob

dir_pl = "/home/thomas/Documents/code/pleiades-boat-detection/data/scoring/diff/PLN/neo"
output_dir = "/home/thomas/Documents/code/pleiades-boat-detection/data/scoring/diff/PLN"
os.makedirs(output_dir, exist_ok=True)

files = glob(os.path.join(dir_pl, "*.tif"))

for f in files:
    out_name = os.path.join(output_dir, os.path.basename(f))
    downsample_tif_x2(f, out_name)