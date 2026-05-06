import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import pyproj
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SAM2MaskGenerator:
    def __init__(self, model_cfg: str, checkpoint_path: str, device: str = "cuda"):
        """Initialise le prédicteur SAM2."""
        logger.info(f"Chargement de SAM2 sur {device}...")
        self.device = device
        self.sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def process_image(
        self, 
        image_path: Path, 
        yolo_obb_path: Path, 
        output_geojson: Path,
        output_crops_dir: Path,  # NEW: Directory for the PNG crops
        padding: int = 128
    ) -> None:
        """Génère des masques et sauvegarde des crops PNG pour chaque objet."""
        
        # 1. Ouverture du raster
        with rasterio.open(image_path) as src:
            src_crs = src.crs
            img_w, img_h = src.width, src.height
            
            # 2. Lecture des OBB (format YOLO)
            obbs = self._read_yolo_obb(yolo_obb_path, img_w, img_h)
            if not obbs:
                logger.warning(f"Aucun OBB trouvé dans {yolo_obb_path.name}")
                return

            transformer_wgs84 = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            features = []

            # Création du dossier pour les crops
            output_crops_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Traitement de {len(obbs)} objets (Génération des masques et des crops)...")

            # 3. Boucle sur chaque bateau détecté
            for idx, (class_id, obb_corners) in enumerate(obbs):
                
                # Emprise globale du bateau
                min_x, min_y = np.min(obb_corners, axis=0)
                max_x, max_y = np.max(obb_corners, axis=0)
                
                # Fenêtre de patch avec marge
                win_min_x = max(0, int(min_x) - padding)
                win_min_y = max(0, int(min_y) - padding)
                win_max_x = min(img_w, int(max_x) + padding)
                win_max_y = min(img_h, int(max_y) + padding)
                
                win_width = win_max_x - win_min_x
                win_height = win_max_y - win_min_y
                
                # 4. Charger le patch
                window = Window(win_min_x, win_min_y, win_width, win_height)
                img_patch = src.read([1, 2, 3], window=window)
                img_patch = np.transpose(img_patch, (1, 2, 0))
                
                if img_patch.dtype != np.uint8:
                    patch_max = img_patch.max()
                    if patch_max > 0:
                        img_patch = (img_patch / patch_max * 255).astype(np.uint8)
                    else:
                        img_patch = img_patch.astype(np.uint8)

                # 5. Conversion OBB vers coordonnées locales du patch
                local_obb_corners = obb_corners - np.array([win_min_x, win_min_y])
                loc_x_coords = local_obb_corners[:, 0]
                loc_y_coords = local_obb_corners[:, 1]
                
                local_aabb = np.array([loc_x_coords.min(), loc_y_coords.min(), loc_x_coords.max(), loc_y_coords.max()])
                local_center = np.array([[loc_x_coords.mean(), loc_y_coords.mean()]])
                
                # 6. Inférence SAM2
                self.predictor.set_image(img_patch)
                masks, scores, _ = self.predictor.predict(
                    point_coords=local_center,
                    point_labels=np.array([1]), 
                    box=local_aabb[None, :],
                    multimask_output=False
                )
                
                best_mask = masks[0]
                
                # ==============================================================
                # NEW LOGIC: GENERATE SQUARE CROPS (ORIGINAL + MASKED)
                # ==============================================================
                y_indices, x_indices = np.where(best_mask)
                if len(x_indices) > 0 and len(y_indices) > 0:
                    # Find exact bounding box of the SAM2 mask
                    m_min_x, m_max_x = x_indices.min(), x_indices.max()
                    m_min_y, m_max_y = y_indices.min(), y_indices.max()
                    
                    # Calculate square dimensions with 10px padding on each side (+20 total)
                    w = m_max_x - m_min_x
                    h = m_max_y - m_min_y
                    size = max(w, h) + 20
                    half_size = size // 2
                    
                    # Center of the mask
                    cx = (m_min_x + m_max_x) // 2
                    cy = (m_min_y + m_max_y) // 2
                    
                    patch_h, patch_w = img_patch.shape[:2]
                    
                    # Calculate square coordinates clamped to the loaded patch
                    sq_min_x = max(0, cx - half_size)
                    sq_max_x = min(patch_w, cx + half_size)
                    sq_min_y = max(0, cy - half_size)
                    sq_max_y = min(patch_h, cy + half_size)
                    
                    # Extract the square crop from the RGB patch and the boolean mask
                    crop_img = img_patch[sq_min_y:sq_max_y, sq_min_x:sq_max_x]

                    # 1. FORCER LE TYPE EN BOOLÉEN ICI :
                    crop_mask = best_mask[sq_min_y:sq_max_y, sq_min_x:sq_max_x].astype(bool)

                    # Create RGBA versions (Add an Alpha channel for transparency)
                    alpha_channel = np.full(crop_img.shape[:2], 255, dtype=np.uint8)
                    rgba_crop = np.dstack((crop_img, alpha_channel))

                    # Create the masked version (Pixels outside the mask become 100% transparent)
                    rgba_masked = rgba_crop.copy()

                    # 2. MAINTENANT LE '~' FONCTIONNERA PARFAITEMENT :
                    rgba_masked[~crop_mask, 3] = 0
                    
                    # Concatenate the images side-by-side horizontally
                    side_by_side = np.hstack((rgba_crop, rgba_masked))
                    
                    # Save to the specified output folder
                    out_png_path = output_crops_dir / f"{image_path.stem}_obj_{idx}.png"
                    Image.fromarray(side_by_side).save(out_png_path)
                # ==============================================================

                mask_polygons = self._mask_to_polygons(best_mask)
                window_transform = src.window_transform(window)
                
                for poly_px in mask_polygons:
                    poly_wgs84 = self._pixel_to_wgs84(poly_px, window_transform, transformer_wgs84)
                    
                    if poly_wgs84 and poly_wgs84.is_valid:
                        features.append({
                            "type": "Feature",
                            "geometry": mapping(poly_wgs84),
                            "properties": {
                                "feature_id": idx,
                                "class_id": class_id,
                                "sam2_score": float(scores[0]),
                                "source_image": image_path.stem
                            }
                        })

        # 8. Sauvegarde GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        output_geojson.parent.mkdir(parents=True, exist_ok=True)
        with open(output_geojson, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)
            
        logger.info(f"Terminé ! GeoJSON : {output_geojson} | Crops PNG sauvés dans : {output_crops_dir}")

    def _read_yolo_obb(self, txt_path: Path, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
        obbs = []
        if not txt_path.exists():
            return obbs
            
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 9:
                    continue
                class_id = int(parts[0])
                corners_norm = np.array(parts[1:], dtype=np.float32).reshape(4, 2)
                corners_px = corners_norm * np.array([img_w, img_h])
                obbs.append((class_id, corners_px))
        return obbs

    def _mask_to_polygons(self, mask: np.ndarray) -> List[Polygon]:
        mask_uint8 = (mask * 255).astype(np.uint8)
        polygons = []
        for geom_dict, value in shapes(mask_uint8, mask=(mask_uint8 > 0)):
            try:
                poly = shape(geom_dict)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.area > 5:
                    polygons.append(poly)
            except Exception as e:
                logger.debug(f"Erreur de création de polygone : {e}")
        return polygons

    def _pixel_to_wgs84(
        self, 
        poly_px: Polygon, 
        affine: rasterio.Affine, 
        transformer: pyproj.Transformer
    ) -> Polygon:
        def px_to_crs(x, y):
            return affine * (x, y)
        poly_crs = transform(px_to_crs, poly_px)
        def crs_to_wgs84(x, y):
            return transformer.transform(x, y)
        poly_wgs84 = transform(crs_to_wgs84, poly_crs)
        return poly_wgs84


if __name__ == "__main__":
    MODEL_CFG = "sam2_hiera_l.yaml"
    CHECKPOINT = "weights/sam2_hiera_large.pt"
    
    img_file = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/dataset/images/train/Nosy_boraha_south.tif")
    obb_file = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/dataset/labels/train/Nosy_boraha_south.txt")
    out_geojson = Path("/home/thomas/Documents/code/pleiades-boat-detection/masks/test1.geojson")
    
    # NEW: Specify your output directory for PNG crops
    out_crops_dir = Path("/home/thomas/Documents/code/pleiades-boat-detection/masks/all")
    
    generator = SAM2MaskGenerator(MODEL_CFG, CHECKPOINT)
    generator.process_image(img_file, obb_file, out_geojson, out_crops_dir)