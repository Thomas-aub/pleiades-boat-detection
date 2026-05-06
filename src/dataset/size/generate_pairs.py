#!/usr/bin/env python3

import os
import random
import csv # Added for CSV export
from glob import glob
import numpy as np
import rasterio
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import AutoModel, PreTrainedModel
from piq import FID # Using FID from PIQ to calculate Fréchet Distance
import matplotlib.pyplot as plt

# --- Configuration ---
# Update this path to where your 13 Pléiades Néo images are located
DIR_PLN = "/home/thomas/Documents/code/pleiades-boat-detection/data/raw"

TILE_SIZE = 640
NB_FEATURES = 500  # Number of random tiles to sample per image

SATDINO_MODEL = "strakajk/satdino-vit_base-16"
FMOW_MEAN = (0.3826, 0.3525, 0.3051)
FMOW_STD  = (0.1598, 0.1474, 0.1384)


# --- Monkey-patch for transformers ---
def _get_all_tied_weights_keys(self):
    val = getattr(self, "_tied_weights_keys", None)
    return val if val is not None else {}
PreTrainedModel.all_tied_weights_keys = property(_get_all_tied_weights_keys)


# --- Utility Functions ---
def read_image(path):
    with rasterio.open(path) as src:
        image = src.read()
    return image[:3].transpose(1, 2, 0)

def delete_non_data(tile, threshold=0.05):
    no_data_pixels = np.all(tile == 0, axis=-1)
    return np.mean(no_data_pixels) < threshold

def is_single_color(tile, tolerance=5):
    min_vals = np.min(tile, axis=(0, 1)).astype(np.int16)
    max_vals = np.max(tile, axis=(0, 1)).astype(np.int16)
    return np.all((max_vals - min_vals) < tolerance)

def cut_into_tiles(image, tile_size=128):
    tiles = []
    for i in range(0, image.shape[0], tile_size):
        for j in range(0, image.shape[1], tile_size):
            tile = image[i:i+tile_size, j:j+tile_size, :]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                if delete_non_data(tile) and not is_single_color(tile):
                    tiles.append(tile)
    return np.array(tiles)

def build_satdino_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=FMOW_MEAN, std=FMOW_STD),
    ])

def tile_to_rgb_uint8(tile):
    rgb = tile[:3].astype(np.float32) if tile.shape[0] >= 3 else np.stack([tile[0].astype(np.float32)] * 3)
    stretched = np.zeros((3,) + rgb.shape[1:], dtype=np.uint8)
    for c in range(3):
        band = rgb[c]
        p2, p98 = np.percentile(band, 0), np.percentile(band, 100)
        if p98 > p2:
            band = (band - p2) / (p98 - p2)
        else:
            band = band - band.min()
            if band.max() > 0:
                band = band / band.max()
        stretched[c] = np.clip(band * 255, 0, 255).astype(np.uint8)
    return np.transpose(stretched, (1, 2, 0))

def extract_satdino_features(tiles, model, transform, device, batch_size=32):
    all_features = []
    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start:start + batch_size]
        tensors = [transform(Image.fromarray(tile_to_rgb_uint8(t), "RGB")) for t in batch_tiles]
        batch_tensor = torch.stack(tensors).to(device)

        with torch.inference_mode():
            features = model(batch_tensor)

        all_features.append(features.float().cpu().numpy())

    return np.concatenate(all_features, axis=0)

# --- Main Execution ---
if __name__ == "__main__":
    list_pln = sorted(glob(os.path.join(DIR_PLN, "**/*.tif"), recursive=True))
    
    if len(list_pln) != 13:
        print(f"Warning: Found {len(list_pln)} TIFF files, expected 13. Processing what was found.")

    print("--- 1. Tiling Images ---")
    datasets_tiles = {}
    for path in list_pln:
        base_name = os.path.splitext(os.path.basename(path))[0]
        tiles = cut_into_tiles(read_image(path), TILE_SIZE)
        
        # Sample immediately to save memory
        if len(tiles) > NB_FEATURES:
            sampled_tiles = random.sample(list(tiles), NB_FEATURES)
        else:
            sampled_tiles = tiles
            print(f"Warning: {base_name} only had {len(tiles)} valid tiles (requested {NB_FEATURES}).")
            
        datasets_tiles[base_name] = sampled_tiles

    print("\n--- 2. Extracting SatDINO Features ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(SATDINO_MODEL, trust_remote_code=True).to(device).eval()
    transform = build_satdino_transform()
    
    datasets_features = {}
    for name, tiles in datasets_tiles.items():
        datasets_features[name] = extract_satdino_features(tiles, model, transform, device, batch_size=64)
        print(f"Extracted {len(datasets_features[name])} features for {name}")

    print("\n--- 3. Calculating Pairwise FKD (FID) Matrix ---")
    metric = FID()
    dataset_names = list(datasets_features.keys())
    nb_datasets = len(dataset_names)
    
    fkd_results = np.zeros((nb_datasets, nb_datasets))
    pairwise_csv_data = [] # List to store data for CSV
    
    for i in range(nb_datasets):
        name_i = dataset_names[i]
        
        # UPDATED: Start j from i to avoid redundant A->B and B->A calculations
        for j in range(i, nb_datasets):
            name_j = dataset_names[j]
            
            if i == j:
                fkd_results[i, j] = 0.0 # Distance to itself is 0
                continue
            
            # The PIQ FID metric expects inputs of shape (N, D)
            x_feats = torch.from_numpy(datasets_features[name_i]).float()
            y_feats = torch.from_numpy(datasets_features[name_j]).float()
            
            # PIQ FID returns a tensor, we need the item
            result = metric(x_feats, y_feats)
            fkd_val = result.item()
            
            # UPDATED: Mirror the result symmetrically in the matrix
            fkd_results[i, j] = fkd_val
            fkd_results[j, i] = fkd_val
            
            # Store the unique pair for CSV export
            pairwise_csv_data.append([name_i, name_j, fkd_val])
            
            print(f"FKD between {name_i} and {name_j}: {fkd_val:.2f}")

    print("\n--- 4. Saving Scores to CSV ---")
    # Sort the list by FKD distance (lowest/most similar first)
    pairwise_csv_data.sort(key=lambda x: x[2])
    
    csv_filename = "fkd_pairwise_scores.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image_A", "Image_B", "FKD_Score"]) # Header
        writer.writerows(pairwise_csv_data)
        
    print(f"Successfully saved {len(pairwise_csv_data)} unique pairs to '{csv_filename}'")


    print("\n--- 5. Generating Sorted Heatmap ---")
    
    # Calculate average distance for each image (ignoring the 0 distance to itself)
    avg_distances = []
    for i in range(nb_datasets):
        # Mask out the diagonal element (which is 0) to get a true average distance to *others*
        mask = np.ones(nb_datasets, dtype=bool)
        mask[i] = False
        avg_dist = np.mean(fkd_results[i, mask])
        avg_distances.append(avg_dist)
        
    # Sort names and matrix based on average distance (closest to others -> farthest from others)
    sort_indices = np.argsort(avg_distances)
    sorted_names = [dataset_names[idx] for idx in sort_indices]
    
    # Re-arrange the rows and columns of the matrix according to the sort
    sorted_matrix = fkd_results[sort_indices, :]
    sorted_matrix = sorted_matrix[:, sort_indices]

    # Plot
    plt.figure(figsize=(14, 10))
    # Use a colormap where lower values (closer) are visually distinct from higher values (farther)
    im = plt.imshow(sorted_matrix, cmap="magma_r", vmin=np.min(sorted_matrix), vmax=np.max(sorted_matrix))
    plt.colorbar(im, label="Fréchet Distance (FKD)")
    
    plt.xticks(range(nb_datasets), sorted_names, rotation=45, ha="right")
    plt.yticks(range(nb_datasets), sorted_names)
    
    # Add numerical values to the heatmap cells
    threshold = np.max(sorted_matrix) / 2.
    for i in range(nb_datasets):
        for j in range(nb_datasets):
            color = "white" if sorted_matrix[i, j] < threshold else "black"
            plt.text(j, i, f"{sorted_matrix[i, j]:.1f}", ha="center", va="center", color=color, fontsize=8)
            
    plt.title("Pairwise FKD Distance Between Pléiades Néo Images\n(Sorted by Average Closeness)")
    plt.tight_layout()
    plt.savefig("PlN_Pairwise_FKD_Sorted.png", dpi=300)
    print("\nPlot saved as 'PlN_Pairwise_FKD_Sorted.png'")