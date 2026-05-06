import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from pathlib import Path

def compute_and_save_tsne(csv_path: Path, pca_path: Path, png_path: Path, folder_name: str) -> None:
    """Computes 2D t-SNE, updates the CSV, and saves a static plot for a given folder."""
    print(f"\n[{folder_name}] Loading data for t-SNE...")
    df = pd.read_csv(csv_path)
    X_pca = np.load(pca_path)

    if len(df) != X_pca.shape[0]:
        print(f"[{folder_name}] Row mismatch: CSV ({len(df)}) vs PCA ({X_pca.shape[0]}). Skipping.")
        return
    
    # Add a small amount of jitter to prevent issues with identical points
    X_pca += np.random.normal(0, 1e-5, X_pca.shape)
    
    print(f"[{folder_name}] Computing t-SNE (this might take a minute)...")
    # Dynamically adjust perplexity if a folder has very few tiles
    perplexity = min(30, len(X_pca) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X_pca)
    
    # Add the new t-SNE coordinates to the DataFrame
    df['tsne_1'] = X_2d[:, 0]
    df['tsne_2'] = X_2d[:, 1]
    
    # Overwrite the CSV with the updated DataFrame
    df.to_csv(csv_path, index=False)
    
    # Generate and save a static t-SNE plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df['cluster'], cmap='tab20', s=15, alpha=0.8)
    plt.title(f"Static t-SNE Visualization: {folder_name}", fontsize=14)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True, alpha=0.3)
    plt.savefig(png_path)
    plt.close()
    
    print(f"[{folder_name}] Success! Updated CSV and saved static plot to {png_path}")

if __name__ == "__main__":
    BASE_DIR = Path("data/scoring/tiled")
    FOLDERS_TO_PROCESS = ["Gen", "PL", "PLN"]

    for folder in FOLDERS_TO_PROCESS:
        folder_path = BASE_DIR / folder
        csv_file = folder_path / "clustering_results.csv"
        pca_file = folder_path / "X_pca.npy"
        png_file = folder_path / "tsne_static.png"

        if csv_file.exists() and pca_file.exists():
            compute_and_save_tsne(csv_file, pca_file, png_file, folder)
        else:
            print(f"\nSkipping {folder}: Missing clustering_results.csv or X_pca.npy")