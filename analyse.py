import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Les deux fichiers GeoJSON à fusionner et analyser
FILES = [
    "predictions/results/Nosy_boraha_south_eval.geojson",
    "predictions/results/IMG_PNEO3_STD_202308070701158_PAN_ORT_PWOI_000373512_15_2_F_1_P_R2C1_eval.geojson"
]

def analyze_thresholds(target_class: str = "Pirogue"):
    tp_confidences = []
    fp_confidences = []
    base_fn_count = 0

    # 1. Collecte des données depuis les GeoJSON
    for filepath in FILES:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️ Attention : le fichier '{filepath}' est introuvable. Il sera ignoré.")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            
            # --- FILTRE DE CLASSE ---
            # On ignore tout ce qui n'est pas la classe ciblée
            if props.get("class_name", "").lower() != target_class.lower():
                continue
            
            # On ignore les prédictions jetées par le post-processing (Coastline / Buildings)
            if props.get("deleted", False):
                continue
                
            label = props.get("label")
            conf = props.get("confidence")
            
            if label == "FN":
                # Les pirogues que le modèle n'a JAMAIS vues
                base_fn_count += 1
            elif label == "TP" and conf is not None:
                tp_confidences.append(conf)
            elif label == "FP" and conf is not None:
                fp_confidences.append(conf)

    # 2. Simulation pour chaque seuil de 0.05 à 0.50
    results = []
    thresholds = np.arange(0.01, 0.51, 0.01)
    
    # Listes pour alimenter les graphiques Matplotlib
    plot_thresholds = []
    plot_precisions = []
    plot_recalls = []
    plot_f1s = []

    for t in thresholds:
        current_tp = sum(1 for c in tp_confidences if c >= t)
        current_fp = sum(1 for c in fp_confidences if c >= t)
        current_fn = base_fn_count + sum(1 for c in tp_confidences if c < t)
        
        total = current_tp + current_fp + current_fn
        
        pct_tp = (current_tp / total * 100) if total > 0 else 0.0
        pct_fp = (current_fp / total * 100) if total > 0 else 0.0
        pct_fn = (current_fn / total * 100) if total > 0 else 0.0
            
        precision = current_tp / (current_tp + current_fp) if (current_tp + current_fp) > 0 else 0.0
        recall = current_tp / (current_tp + current_fn) if (current_tp + current_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
        results.append({
            "Seuil": f"{t:.2f}",
            "TP": current_tp,
            "FP": current_fp,
            "FN": current_fn,
            "% TP": f"{pct_tp:.1f}%",
            "% FP": f"{pct_fp:.1f}%",
            "% FN": f"{pct_fn:.1f}%",
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1_score, 4)
        })
        
        plot_thresholds.append(t)
        plot_precisions.append(precision)
        plot_recalls.append(recall)
        plot_f1s.append(f1_score)

    # 3. Affichage et Export CSV
    if not results:
        print(f"❌ Aucune donnée à analyser pour la classe '{target_class}'.")
        return

    df = pd.DataFrame(results)
    
    print(f"\n📊 ANALYSE DES SEUILS DE CONFIANCE ({target_class.upper()}) :")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)
    
    out_csv = f"analyse_{target_class.lower()}.csv"
    df.to_csv(out_csv, index=False, sep=",")
    print(f"\n✅ Fichier CSV généré : {out_csv}")

    # 4. Génération des graphiques
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        pass

    # --- Courbe Recall ---
    plt.figure(figsize=(8, 5))
    plt.plot(plot_thresholds, plot_recalls, marker='o', color='#2196F3', linewidth=2)
    plt.title(f'Recall vs Confidence Threshold ({target_class})', fontsize=14, pad=10)
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'recallXconf_{target_class.lower()}.png', dpi=300)
    plt.close()
    print(f"✅ Graphique généré : recallXconf_{target_class.lower()}.png")

    # --- Courbe Precision ---
    plt.figure(figsize=(8, 5))
    plt.plot(plot_thresholds, plot_precisions, marker='o', color='#4CAF50', linewidth=2)
    plt.title(f'Precision vs Confidence Threshold ({target_class})', fontsize=14, pad=10)
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'precXconf_{target_class.lower()}.png', dpi=300)
    plt.close()
    print(f"✅ Graphique généré : precXconf_{target_class.lower()}.png")

    # --- Courbe F1-Score ---
    plt.figure(figsize=(8, 5))
    plt.plot(plot_thresholds, plot_f1s, marker='o', color='#F44336', linewidth=2)
    plt.title(f'F1-Score vs Confidence Threshold ({target_class})', fontsize=14, pad=10)
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'f1Xconf_{target_class.lower()}.png', dpi=300)
    plt.close()
    print(f"✅ Graphique généré : f1Xconf_{target_class.lower()}.png")

if __name__ == "__main__":
    # Tu peux changer la cible ici si un jour tu veux évaluer 'Small_Motorboat'
    analyze_thresholds(target_class="Pirogue")