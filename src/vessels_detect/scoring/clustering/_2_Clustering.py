import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from typing import Tuple

def run_clustering(
    df: pd.DataFrame, 
    n_clusters: int = 4
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Performs clustering exclusively on image embeddings.
    """
    # 1. Extract just the image features
    X_combined = np.stack(df['img_feature'].values)
    
    # 2. Standardize the feature set
    X_scaled = StandardScaler().fit_transform(X_combined)
    
    # 3. Apply PCA to reduce dimensionality for stable clustering
    n_components = min(50, X_scaled.shape[0]) 
    pca_model = PCA(n_components=n_components, random_state=42)
    X_pca = pca_model.fit_transform(X_scaled)
    
    # 4. Add microscopic jitter to prevent identical embeddings from collapsing the GMM covariance
    # (e.g., if there are completely identical tiles of empty water)
    X_pca += np.random.normal(0, 1e-6, X_pca.shape)
    
    # 5. Store the first principal component (PC1)
    df['pca_1d'] = X_pca[:, 0]
    
    # 6. Perform GMM clustering on the PCA-reduced data
    # IMPORTANT FIX: reg_covar=1e-3 ensures the covariance matrix remains positive-definite
    # even if a cluster has low variance or fewer samples than dimensions.
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, reg_covar=1e-3)
    df['cluster'] = gmm.fit_predict(X_pca)
    
    return df, X_pca

if __name__ == "__main__":
    BASE_DIR = Path("data/scoring/tiled")
    FOLDERS_TO_PROCESS = ["Gen", "PL", "PLN"]
    NUMBER_OF_CLUSTERS = 3  
    
    for folder in FOLDERS_TO_PROCESS:
        folder_path = BASE_DIR / folder
        pkl_file = folder_path / "embeddings.pkl"
        
        if pkl_file.exists():
            print(f"\nRunning Clustering on: {folder}...")
            df = pd.read_pickle(pkl_file)
            
            try:
                # Run clustering 
                df_res, X_reduced = run_clustering(df, n_clusters=NUMBER_OF_CLUSTERS)
                
                # Save Outputs
                result_csv = folder_path / "clustering_results.csv"
                pca_file = folder_path / "X_pca.npy"
                
                df_res.to_csv(result_csv, index=False)
                np.save(pca_file, X_reduced)
                
                print(f"✅ Success! Saved {result_csv.name} and {pca_file.name} in {folder_path}/")
            except Exception as e:
                print(f"❌ Failed clustering on {folder}. Error: {e}")
        else:
            print(f"❌ Error: {pkl_file} not found. Run _1_Embedding.py first.")