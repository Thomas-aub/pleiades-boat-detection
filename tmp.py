import json
from pathlib import Path
from pyproj import Transformer

def fix_geojson_crs(input_path: Path, output_path: Path, src_epsg: int = 32739):
    """
    Convertit les coordonnées d'un GeoJSON de mètres (UTM) vers degrés (WGS84).
    Conserve la précision maximale et le format structurel strict.
    """
    if not input_path.exists():
        print(f"❌ Fichier introuvable : {input_path}")
        return

    # 1. Charger le fichier
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Préparer le transformateur (Mètres UTM -> Degrés WGS84)
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", "EPSG:4326", always_xy=True)

    # 3. Fonction récursive pour traiter les coordonnées imbriquées
    def transform_coords(coords):
        if isinstance(coords[0], (int, float)):
            # C'est une paire [x, y] en mètres
            lon, lat = transformer.transform(coords[0], coords[1])
            # On retourne la valeur brute pour garder les 15 décimales comme dans votre exemple
            return [lon, lat]
        else:
            # C'est une liste d'anneaux ou de points
            return [transform_coords(c) for c in coords]

    # 4. Appliquer la conversion à tous les bateaux
    n_features = 0
    for feature in data.get("features", []):
        geom = feature.get("geometry")
        if geom and "coordinates" in geom:
            geom["coordinates"] = transform_coords(geom["coordinates"])
            n_features += 1

    # 5. Forcer exactement le même bloc "crs" que dans votre exemple
    data["crs"] = {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    }

    # 6. Sauvegarder le fichier avec une indentation propre (2 espaces)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Conversion réussie ! {n_features} polygones transformés.")
    print(f"📂 Sauvegardé dans : {output_path}")

# --- Exécution ---
if __name__ == "__main__":
    # Définition des chemins
    raw_dir = Path("data/raw")
    
    # Mettez ici le nom du fichier qui pose problème (celui avec les coordonnées en mètres)
    nom_fichier = "/home/thomas/Documents/code/pleiades-boat-detection/data/raw/IMG_PNEO3_STD_202305170722489_PAN_ORT_PWOI_000373521_3_2_F_1_P_R4C1"
    
    fichier_entree = raw_dir / f"{nom_fichier}.geojson"
    fichier_sortie = raw_dir / f"{nom_fichier}.geojson" # Écrase le fichier d'origine
    
    # On lance la conversion
    fix_geojson_crs(fichier_entree, fichier_sortie, src_epsg=32739)