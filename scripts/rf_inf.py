import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import supervision as sv

def xyxy_to_yolo_norm(xyxy, img_w, img_h):
    """Convertit [x_min, y_min, x_max, y_max] absolus vers [x_center, y_center, w, h] normalisés."""
    x_center = ((xyxy[0] + xyxy[2]) / 2) / img_w
    y_center = ((xyxy[1] + xyxy[3]) / 2) / img_h
    width = (xyxy[2] - xyxy[0]) / img_w
    height = (xyxy[3] - xyxy[1]) / img_h
    return x_center, y_center, width, height

def load_yolo_predictions(txt_path, img_size=1536):
    """Charge les prédictions (format HBB) depuis un .txt vers un objet supervision."""
    if not txt_path.exists():
        return sv.Detections.empty()
    
    boxes, confidences, class_ids = [], [], []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:  # class cx cy w h conf
                c, cx, cy, w, h, conf = map(float, parts[:6])
                # Repassage en coordonnées absolues pour Supervision
                x_min = (cx - w / 2) * img_size
                y_min = (cy - h / 2) * img_size
                x_max = (cx + w / 2) * img_size
                y_max = (cy + h / 2) * img_size
                
                boxes.append([x_min, y_min, x_max, y_max])
                confidences.append(conf)
                class_ids.append(0) # On force la classe 0 (Bateau)
                
    if not boxes:
        return sv.Detections.empty()
        
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confidences, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int)
    )

def load_yolo_gt(txt_path, img_size=1536):
    """Charge la vérité terrain (format OBB ou HBB) depuis un .txt vers un objet supervision."""
    if not txt_path.exists():
        return sv.Detections.empty()
        
    boxes, class_ids = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) == 5:  # GT est en HBB : class cx cy w h
                c, cx, cy, w, h = map(float, parts)
                x_min = (cx - w / 2) * img_size
                y_min = (cy - h / 2) * img_size
                x_max = (cx + w / 2) * img_size
                y_max = (cy + h / 2) * img_size
                boxes.append([x_min, y_min, x_max, y_max])
                class_ids.append(0)
                
            elif len(parts) == 9:  # GT est en OBB : class x1 y1 x2 y2 x3 y3 x4 y4
                coords = np.array(parts[1:], dtype=float).reshape(4, 2)
                coords *= img_size
                x_min, y_min = np.min(coords[:, 0]), np.min(coords[:, 1])
                x_max, y_max = np.max(coords[:, 0]), np.max(coords[:, 1])
                boxes.append([x_min, y_min, x_max, y_max])
                class_ids.append(0) # On force la classe 0 (Bateau)

    if not boxes:
        return sv.Detections.empty()
        
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int)
    )

def evaluate_predictions(pred_dir: Path, gt_dir: Path, iou_threshold: float = 0.20):
    """Compare les prédictions aux vérités terrains et affiche les métriques."""
    print(f"\n[{'='*40}]")
    print(f"LANCEMENT DE L'ÉVALUATION (IoU = {iou_threshold})")
    print(f"[{'='*40}]")
    
    all_preds = []
    all_gts = []
    
    # On rassemble tous les noms de fichiers uniques (au cas où il y a des faux négatifs stricts)
    all_stems = set(p.stem for p in pred_dir.glob("*.txt")) | set(p.stem for p in gt_dir.glob("*.txt"))
    
    if not all_stems:
        print("Erreur : Aucun fichier .txt trouvé pour l'évaluation.")
        return

    for stem in all_stems:
        pred_path = pred_dir / f"{stem}.txt"
        gt_path = gt_dir / f"{stem}.txt"
        
        preds = load_yolo_predictions(pred_path)
        gts = load_yolo_gt(gt_path)
        
        all_preds.append(preds)
        all_gts.append(gts)

    # Matrice de confusion pour P, R, F1 avec IoU spécifique
    cm = sv.ConfusionMatrix.from_detections(
        predictions=all_preds,
        targets=all_gts,
        classes=[0],
        iou_threshold=iou_threshold
    )
    
    metrics = cm.matrix
    tp = np.diag(metrics)[0]
    fp = metrics.sum(axis=0)[0] - tp
    fn = metrics.sum(axis=1)[0] - tp
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # mAP classique (Supervision gère le mAP@50 via une courbe P-R standard)
    map_metric = sv.MeanAveragePrecision.from_detections(
        predictions=all_preds,
        targets=all_gts,
    )
    
    print(f"Images analysées  : {len(all_stems)}")
    print(f"Vrais Positifs    : {int(tp)}")
    print(f"Faux Positifs     : {int(fp)}")
    print(f"Faux Négatifs     : {int(fn)}")
    print("-" * 42)
    print(f"mAP@50            : {map_metric.map50:.4f}")
    print(f"Précision (@0.20) : {precision:.4f}")
    print(f"Rappel    (@0.20) : {recall:.4f}")
    print(f"F1-Score  (@0.20) : {f1_score:.4f}")
    print(f"[{'='*40}]\n")

def main():
    # --- Configuration ---
    images_dir = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/dataset/tiled/images/train")
    gt_dir = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/dataset/tiled/labels/train")
    output_dir = Path("predicted")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    COCO_BOAT_ID = 8
    CONFIDENCE_THRESHOLD = 0.3
    
    # --- Chargement du modèle ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement de RF-DETR-Medium sur {device}...")
    processor = AutoImageProcessor.from_pretrained("Roboflow/rf-detr-medium")
    model = AutoModelForObjectDetection.from_pretrained("Roboflow/rf-detr-medium").to(device)
    
    image_paths = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.tiff"))
    print(f"Début de l'inférence sur {len(image_paths)} images...\n")

    for img_path in image_paths:
        try:
            # Lecture de l'image
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size
            
            # Préparation et Inférence
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Post-processing
            target_sizes = torch.tensor([image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONFIDENCE_THRESHOLD
            )[0]
            
            # Écriture des résultats
            out_file = output_dir / f"{img_path.stem}.txt"
            boats_found = 0
            
            with open(out_file, "w") as f:
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if label.item() == COCO_BOAT_ID:
                        boats_found += 1
                        # Conversion
                        box_np = box.cpu().numpy()
                        cx, cy, bw, bh = xyxy_to_yolo_norm(box_np, img_w, img_h)
                        conf = score.item()
                        
                        # Écriture au format YOLO HBB : class(0) cx cy w h conf
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.4f}\n")
                        
            print(f"[{boats_found:^3} bateaux] -> {out_file.name}")
            
        except Exception as e:
            print(f"Erreur sur l'image {img_path.name} : {e}")

    print(f"\nTerminé ! Les prédictions sont dans '{output_dir.absolute()}'")
    
    # ---------------------------------------------------------
    # ÉVALUATION
    # ---------------------------------------------------------
    evaluate_predictions(pred_dir=output_dir, gt_dir=gt_dir, iou_threshold=0.20)

if __name__ == "__main__":
    main()