"""
src/vessels_detect/inference/filters.py
-------------------------
Geospatial filtering for post-NMS detections.
"""
import logging
from typing import List
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from src.vessels_detect.inference.postprocess import GlobalDetection

logger = logging.getLogger(__name__)

class CoastlineFilter:
    """Filters detections that fall outside a valid maritime zone.
    
    Args:
        valid_area_path: Path to a vector file (GeoJSON, GPKG, Shapefile) 
            representing the valid sea area + 100m buffer.
    """
    def __init__(self, valid_area_path: Path):
        if not valid_area_path.exists():
            raise FileNotFoundError(f"Mask file not found: {valid_area_path}")
            
        logger.info("Loading valid maritime area from: %s", valid_area_path)
        # Load the mask and explicitly store its CRS
        self.valid_area = gpd.read_file(valid_area_path)
        self.mask_crs = self.valid_area.crs

    def run(self, detections: List[GlobalDetection]) -> List[GlobalDetection]:
        if not detections:
            return []

        # 1. Convert detections into a GeoDataFrame for vectorised operations
        geometries = [Polygon(det.crs_corners) for det in detections]
        
        # We assume all detections in this batch share the same CRS 
        # (true if they come from the same tile/region)
        det_crs = detections[0].crs
        
        gdf_dets = gpd.GeoDataFrame(
            {"index": range(len(detections))}, 
            geometry=geometries, 
            crs=det_crs
        )

        # 2. Align CRS: Project detections to the mask's CRS if they differ
        if gdf_dets.crs != self.mask_crs:
            gdf_dets = gdf_dets.to_crs(self.mask_crs)

        # 3. Spatial Filter: Keep detections that intersect the valid maritime area
        # 'predicate="intersects"' means if even 1 pixel touches the valid area, we keep it. 
        # You can change to "within" if the whole boat must be in the water.
        valid_gdf = gpd.sjoin(gdf_dets, self.valid_area, how="inner", predicate="intersects")
        
        # 4. Reconstruct the filtered list
        valid_indices = set(valid_gdf["index"].tolist())
        filtered_detections = [
            det for i, det in enumerate(detections) if i in valid_indices
        ]

        logger.info(
            "Coastline filter: kept %d / %d detections.", 
            len(filtered_detections), len(detections)
        )
        return filtered_detections