import cv2
import torch
import numpy as np
import rasterio
from pathlib import Path
from shapely.geometry import Polygon

# Import spécifique à SAM 3 dans Ultralytics
from ultralytics.models.sam import SAM3SemanticPredictor

# ---------------------------------------------------------
# UTILITAIRES D'IMAGE ET DE FORMATAGE
# ---------------------------------------------------------

def load_and_stretch_tiff(tiff_path):
    """Charge un GeoTIFF 16-bit et l'étire en 8-bit BGR visible pour le modèle."""
    with rasterio.open(tiff_path) as src:
        img_array = src.read()
        
        if img_array.shape[0] >= 3:
            img_array = img_array[:3, :, :]
        elif img_array.shape[0] == 1:
            img_array = np.repeat(img_array, 3, axis=0)
            
        img_array = np.transpose(img_array, (1, 2, 0))
        valid_pixels = img_array[img_array > 0]
        
        if len(valid_pixels) == 0:
            return np.zeros_like(img_array, dtype=np.uint8), src.width, src.height
            
        p2, p98 = np.percentile(valid_pixels, (1.0, 99.0))
        img_stretched = np.clip((img_array - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255)
        img_bgr = cv2.cvtColor(img_stretched.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        return img_bgr, src.width, src.height

def mask_to_obb_norm(mask_np, img_w, img_h):
    """Convertit un masque binaire en coordonnées OBB normalisées (8 points)."""
    contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5: 
        return None
        
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect) 
    
    box[:, 0] /= img_w
    box[:, 1] /= img_h
    box = np.clip(box, 0.0, 1.0)
    return box

# ---------------------------------------------------------
# LOGIQUE D'ÉVALUATION POLYGONALE (SHAPELY) AVEC MAP
# ---------------------------------------------------------

def load_polygons_from_txt(txt_path, has_confidence=False):
    """Charge des OBB YOLO en objets Polygon Shapely."""
    polys = []
    if not txt_path.exists():
        return polys
        
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if (not has_confidence and len(parts) == 9) or (has_confidence and len(parts) == 10):
                coords = np.array(parts[1:9], dtype=float).reshape(4, 2)
                poly = Polygon(coords * 1000)
                if poly.is_valid and poly.area > 0:
                    if has_confidence:
                        polys.append({'poly': poly, 'conf': float(parts[9])})
                    else:
                        polys.append(poly)
    return polys

def compute_ap(recall, precision):
    """Calcule l'Average Precision en utilisant l'interpolation sur tous les points (méthode COCO/Pascal VOC)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Rendre la courbe de précision monotone décroissante
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    # Calculer l'aire sous la courbe
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

def calculate_strict_metrics(pred_dir, gt_dir, base_iou_threshold=0.20):
    """Évaluation stricte des OBB : calcule P/R/F1 à un IoU de base, et le mAP@50 global."""
    print(f"\n[{'='*50}]")
    print(f"ÉVALUATION STRICTE OBB (Calcul géométrique en cours...)")
    print(f"[{'='*50}]")
    
    all_stems = set(p.stem for p in pred_dir.glob("*.txt")) | set(p.stem for p in gt_dir.glob("*.txt"))
    
    all_gts = {}
    all_preds = []
    total_gt = 0

    # 1. Chargement de toutes les données en mémoire
    for stem in all_stems:
        gt_path = gt_dir / f"{stem}.txt"
        pred_path = pred_dir / f"{stem}.txt"
        
        gts = load_polygons_from_txt(gt_path, has_confidence=False)
        all_gts[stem] = gts
        total_gt += len(gts)
        
        preds = load_polygons_from_txt(pred_path, has_confidence=True)
        for p in preds:
            p['stem'] = stem  # On garde la trace de l'image d'origine
            all_preds.append(p)

    # 2. Évaluation à Seuil Fixe (IoU = 0.20, Conf = 0.20)
    print(f"Calcul Précision/Rappel (IoU={base_iou_threshold}) ...")
    total_tp_base = 0
    total_fp_base = 0
    
    for stem in all_stems:
        gts = all_gts[stem]
        preds = [p for p in all_preds if p['stem'] == stem and p['conf'] >= 0.20]
        preds.sort(key=lambda x: x['conf'], reverse=True)
        
        matched_gt = set()
        for pred in preds:
            best_iou = 0.0
            best_gt_idx = -1
            pred_poly = pred['poly']
            
            for gt_idx, gt_poly in enumerate(gts):
                if gt_idx in matched_gt:
                    continue
                inter_area = pred_poly.intersection(gt_poly).area
                if inter_area > 0:
                    iou = inter_area / pred_poly.union(gt_poly).area
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= base_iou_threshold:
                total_tp_base += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp_base += 1
                
    total_fn_base = total_gt - total_tp_base
    precision_base = total_tp_base / (total_tp_base + total_fp_base + 1e-6)
    recall_base = total_tp_base / (total_tp_base + total_fn_base + 1e-6)
    f1_base = 2 * (precision_base * recall_base) / (precision_base + recall_base + 1e-6)

    # 3. Calcul du mAP@50 (IoU = 0.50, toutes confiances confondues)
    print(f"Calcul mAP@50 (IoU=0.50) ...")
    all_preds.sort(key=lambda x: x['conf'], reverse=True)
    
    tp_array = np.zeros(len(all_preds))
    fp_array = np.zeros(len(all_preds))
    matched_gts_map = {stem: set() for stem in all_gts.keys()}

    for i, pred in enumerate(all_preds):
        stem = pred['stem']
        pred_poly = pred['poly']
        gts = all_gts[stem]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_poly in enumerate(gts):
            if gt_idx in matched_gts_map[stem]:
                continue
            inter_area = pred_poly.intersection(gt_poly).area
            if inter_area > 0:
                iou = inter_area / pred_poly.union(gt_poly).area
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
        if best_iou >= 0.50:
            tp_array[i] = 1
            matched_gts_map[stem].add(best_gt_idx)
        else:
            fp_array[i] = 1

    acc_fp = np.cumsum(fp_array)
    acc_tp = np.cumsum(tp_array)
    
    recalls = acc_tp / total_gt if total_gt > 0 else np.zeros_like(acc_tp)
    precisions = acc_tp / (acc_tp + acc_fp + 1e-16)
    
    map50 = compute_ap(recalls, precisions)

    # --- AFFICHAGE ---
    print(f"\nImages analysées  : {len(all_stems)}")
    print(f"Total Vérités Terrain (Bateaux) : {total_gt}")
    print("-" * 50)
    print(f"RÉSULTATS À SEUIL FIXE (IoU >= {base_iou_threshold}, Conf >= 0.20):")
    print(f"  Vrais Positifs    : {total_tp_base}")
    print(f"  Faux Positifs     : {total_fp_base}")
    print(f"  Faux Négatifs     : {total_fn_base}")
    print(f"  Précision         : {precision_base:.4f}")
    print(f"  Rappel            : {recall_base:.4f}")
    print(f"  F1-Score          : {f1_base:.4f}")
    print("-" * 50)
    print(f"MÉTRIQUE GLOBALE :")
    print(f"  mAP@50            : {map50:.4f}")
    print(f"[{'='*50}]\n")


# ---------------------------------------------------------
# SCRIPT PRINCIPAL
# ---------------------------------------------------------

def main():
    images_dir = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/dataset/tiled/images/train")
    gt_dir = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/dataset/tiled/labels/train")
    output_dir = Path("predicted")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ⚠️ Pour un calcul mAP PARFAIT, on laisse générer beaucoup de boîtes (conf très basse)
    # Plus on a de boîtes basse-confiance, plus la courbe mAP sera précise.
    CONFIDENCE_THRESHOLD = 0.05 
    TEXT_PROMPT = ["boat"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de SAM 3 sur {device}...")
    
    overrides = dict(
        model="/home/thomas/Documents/code/pleiades-boat-detection/weights/sam3.pt",
        conf=CONFIDENCE_THRESHOLD,
        device=device,
        save=False,
        verbose=False,
        retina_masks=True # OBLIGATOIRE : Force la génération de masques HD
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    
    image_paths = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.tiff"))
    print(f"Début de l'inférence sur {len(image_paths)} images...\n")

    for img_path in image_paths:
        try:
            # 1. Chargement et étirement du TIFF Pléiades
            img_bgr, img_w, img_h = load_and_stretch_tiff(img_path)
            
            # 2. Inférence SAM 3
            predictor.set_image(img_bgr)
            results = predictor(text=TEXT_PROMPT)
            result = results[0]
            
            out_file = output_dir / f"{img_path.stem}.txt"
            boats_found = 0
            
            with open(out_file, "w") as f:
                # 3. Récupération des Masques
                if result.masks is not None:
                    masks_np = result.masks.data.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for mask_idx, mask in enumerate(masks_np):
                        conf = confs[mask_idx]
                        
                        # 4. Conversion du Masque en OBB
                        obb_coords = mask_to_obb_norm(mask, result.orig_shape[1], result.orig_shape[0])
                        
                        if obb_coords is not None:
                            boats_found += 1
                            flat_coords = obb_coords.flatten()
                            coords_str = " ".join([f"{v:.6f}" for v in flat_coords])
                            
                            # Écriture OBB YOLO : class x1 y1 x2 y2 x3 y3 x4 y4 conf
                            f.write(f"0 {coords_str} {conf:.4f}\n")
                            
            print(f"[{boats_found:^3} bateaux] -> {out_file.name}")
            
        except Exception as e:
            print(f"Erreur sur l'image {img_path.name} : {e}")

    print(f"\nTerminé ! Les prédictions OBB sont dans '{output_dir.absolute()}'")
    
    # 5. Évaluation stricte Shapely (calcule P/R@0.20 ET mAP@50)
    calculate_strict_metrics(pred_dir=output_dir, gt_dir=gt_dir, base_iou_threshold=0.20)

if __name__ == "__main__":
    main()