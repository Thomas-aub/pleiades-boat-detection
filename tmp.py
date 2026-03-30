import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- 1. Définition des chemins ---
# On cible spécifiquement l'image qui a posé problème
filename = "Nosy_boraha_south.geojson"
preds_path = Path(f"data/eval/predictions/raw/{filename}")
mask_path = Path(f"data/eval/Coastlines/{filename}")

print("="*60)
print(f"🔍 ANALYSE GEOSPATIALE : {filename}")
print("="*60)

# --- 2. Chargement et vérification ---
if not preds_path.exists() or not mask_path.exists():
    print("❌ Fichiers introuvables. Vérifiez les chemins.")
    exit()

gdf_preds = gpd.read_file(preds_path)
gdf_mask = gpd.read_file(mask_path)

# Print CRS explicitly
print("\n🌍 1. VERIFICATION DES CRS :")
print(f"   - Bateaux (Preds) : {gdf_preds.crs}")
print(f"   - Masque (Coast)  : {gdf_mask.crs}")
if gdf_preds.crs != gdf_mask.crs:
    print("   ⚠️ ATTENTION: Les CRS sont différents !")

# Check coordinate ranges
print("\n📏 2. LIMITES SPATIALES (Bounding Boxes) :")
p_minx, p_miny, p_maxx, p_maxy = gdf_preds.total_bounds
m_minx, m_miny, m_maxx, m_maxy = gdf_mask.total_bounds
print(f"   - Bateaux : X[{p_minx:.4f} à {p_maxx:.4f}] | Y[{p_miny:.4f} à {p_maxy:.4f}]")
print(f"   - Masque  : X[{m_minx:.4f} à {m_maxx:.4f}] | Y[{m_miny:.4f} à {m_maxy:.4f}]")

# --- 3. Test Manual Intersection ---
print("\n🧮 3. TEST D'INTERSECTION MATHEMATIQUE :")
# On unifie toutes les géométries pour avoir un seul gros objet de chaque côté
preds_union = gdf_preds.unary_union
mask_union = gdf_mask.unary_union

# Nettoyage au cas où les géométries sont corrompues
if not preds_union.is_valid: preds_union = preds_union.buffer(0)
if not mask_union.is_valid: mask_union = mask_union.buffer(0)

is_intersecting = preds_union.intersects(mask_union)
intersection_area = preds_union.intersection(mask_union).area

print(f"   - Est-ce qu'ils se touchent ? : {is_intersecting}")
print(f"   - Surface d'intersection    : {intersection_area:.8e} degrés carrés")

# --- 4. Tracé (Plotting) ---
print("\n🗺️  4. GENERATION DE LA CARTE...")
fig, ax = plt.subplots(figsize=(10, 10))

# Dessiner le masque (en bleu semi-transparent avec bords noirs)
if not gdf_mask.empty:
    gdf_mask.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5, label='Masque (Coastline)')

# Dessiner les prédictions (en rouge vif)
if not gdf_preds.empty:
    gdf_preds.plot(ax=ax, color='red', markersize=5, label='Bateaux (Preds)')

ax.set_title(f"Superposition Bateaux vs Masque\n{filename}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Création d'une légende propre
import matplotlib.patches as mpatches
mask_patch = mpatches.Patch(color='lightblue', label='Masque (Coastlines)')
pred_patch = mpatches.Patch(color='red', label='Bateaux (Prédictions)')
ax.legend(handles=[mask_patch, pred_patch])

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Afficher l'image
plt.show()