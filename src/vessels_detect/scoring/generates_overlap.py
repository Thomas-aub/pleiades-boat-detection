import logging
from pathlib import Path
from typing import Tuple

import rasterio
from rasterio.mask import mask
from shapely.geometry import box

logger = logging.getLogger(__name__)

def make_overlap(tif_path_1: Path, tif_path_2: Path, out_dir: Path) -> Tuple[Path, Path]:
    """
    Takes two GeoTIFFs, calculates their exact spatial intersection, 
    and crops both to this shared bounding box.
    
    Args:
        tif_path_1: Path to the first GeoTIFF.
        tif_path_2: Path to the second GeoTIFF.
        out_dir: Directory to save the cropped outputs.
        
    Returns:
        Tuple containing the paths to the two new cropped GeoTIFFs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(tif_path_1) as src1, rasterio.open(tif_path_2) as src2:
        # 1. Coordinate Reference System (CRS) Safety Check
        if src1.crs != src2.crs:
            raise ValueError(
                f"CRS mismatch! Image 1 is {src1.crs}, Image 2 is {src2.crs}. "
                "They must be in the same projection to calculate a valid overlap."
            )
            
        # 2. Extract bounding boxes (left, bottom, right, top)
        b1 = src1.bounds
        b2 = src2.bounds
        
        # 3. Calculate the intersection of the two bounding boxes
        # The intersecting box is the MAX of the minimums and the MIN of the maximums.
        inter_left = max(b1.left, b2.left)
        inter_bottom = max(b1.bottom, b2.bottom)
        inter_right = min(b1.right, b2.right)
        inter_top = min(b1.top, b2.top)
        
        # 4. Check if they actually overlap
        if inter_left >= inter_right or inter_bottom >= inter_top:
            raise ValueError(f"No spatial overlap exists between {tif_path_1.name} and {tif_path_2.name}.")
            
        logger.info("Found valid spatial intersection. Cropping...")
        
        # 5. Create a Shapely geometry for the intersection
        intersection_geom = [box(inter_left, inter_bottom, inter_right, inter_top)]
        
        # 6. Crop Image 1
        out_img1, out_transform1 = mask(src1, intersection_geom, crop=True)
        out_meta1 = src1.meta.copy()
        out_meta1.update({
            "height": out_img1.shape[1],
            "width": out_img1.shape[2],
            "transform": out_transform1
        })
        
        out_path1 = out_dir / f"{tif_path_1.stem}_intersect.tif"
        with rasterio.open(out_path1, "w", **out_meta1) as dest1:
            dest1.write(out_img1)
            
        # 7. Crop Image 2
        out_img2, out_transform2 = mask(src2, intersection_geom, crop=True)
        out_meta2 = src2.meta.copy()
        out_meta2.update({
            "height": out_img2.shape[1],
            "width": out_img2.shape[2],
            "transform": out_transform2
        })
        
        out_path2 = out_dir / f"{tif_path_2.stem}_intersect.tif"
        with rasterio.open(out_path2, "w", **out_meta2) as dest2:
            dest2.write(out_img2)
            
    logger.info(f"Cropped images saved to {out_dir}")
    
    # 8. Sanity check: Ensure the resulting pixel grids have identical dimensions
    if out_img1.shape[1:] != out_img2.shape[1:]:
        logger.warning(
            f"Shape mismatch after cropping! Img1: {out_img1.shape[1:]}, Img2: {out_img2.shape[1:]}. "
            "This usually happens if the input images have different resolutions (GSD) or misaligned pixel grids."
        )

    return out_path1, out_path2