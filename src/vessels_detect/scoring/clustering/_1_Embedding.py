import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
import torch
from transformers import AutoModel
import torchvision.transforms as T
import rasterio
from typing import Literal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_embeddings_batch(
    df: pd.DataFrame, 
    model_name: Literal["SatDINO", "ResNet50"] = "SatDINO",
    batch_size: int = 8  # Lowered from 32/64 to handle 640x640/1024x1024 tiles without OOM
) -> pd.DataFrame:
    if model_name == "SatDINO":
        return get_embeddings_batch_SatDINO(df, batch_size)
    elif model_name == "ResNet50":
        return get_embeddings_batch_ResNet50(df, batch_size)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def load_image_with_rasterio(path):
    """Helper to safely load a TIF and return a numpy array."""
    with rasterio.open(path) as src:
        return src.read()

def get_embeddings_batch_SatDINO(df: pd.DataFrame, batch_size: int = 8) -> pd.DataFrame:
    print("--- Initializing SatDINO (ViT-Base-16) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "strakajk/satdino-vit_base-16"

    _orig_linspace = torch.linspace
    def _safe_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return _orig_linspace(*args, **kwargs)
    torch.linspace = _safe_linspace

    try:
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=False, device_map=None
        ).to(device)
    finally:
        torch.linspace = _orig_linspace

    model.eval()

    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df['img_feature'] = None
    df['img_feature'] = df['img_feature'].astype(object)
    
    total_samples = len(df)
    pbar = tqdm(total=total_samples, desc="Generating SatDINO Embeddings", unit="img")

    for i in range(0, total_samples, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_tensors = []
        indices = []
        
        for idx, row in batch_df.iterrows():
            try:
                img = load_image_with_rasterio(row['chip_path'])
                
                if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                    img = np.transpose(img, (1, 2, 0))
                if img.shape[-1] > 3:
                    img = img[..., :3]
                
                if img.dtype == np.uint16:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip(img, p2, p98)
                    img = ((img - p2) / (p98 - p2 + 1e-6) * 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = (img / img.max() * 255).clip(0, 255).astype(np.uint8)
                
                batch_tensors.append(preprocess(img))
                indices.append(idx)
            except Exception as e:
                print(f"ERROR [{idx}] {row['chip_path']}: {e}")
                continue
            
        if batch_tensors:
            input_batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                outputs = model(input_batch)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state
                    embeddings = (hidden[:, 0, :] if hidden.dim() == 3 else hidden).cpu().numpy()
                else:
                    embeddings = (outputs[:, 0, :] if outputs.dim() == 3 else outputs).cpu().numpy()

            for j, real_idx in enumerate(indices):
                df.at[real_idx, 'img_feature'] = embeddings[j]
        
        pbar.update(len(batch_tensors))
        
    pbar.close()
    return df.dropna(subset=['img_feature'])

def get_embeddings_batch_ResNet50(df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    print("--- Initializing ResNet50 ---")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    df['img_feature'] = None
    df['img_feature'] = df['img_feature'].astype(object)
    
    total_samples = len(df)
    pbar = tqdm(total=total_samples, desc="Generating ResNet50 Embeddings", unit="img")

    for i in range(0, total_samples, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        imgs, indices = [], []
        
        for idx, row in batch_df.iterrows():
            try:
                img = load_image_with_rasterio(row['chip_path'])
                
                if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                    img = np.transpose(img, (1, 2, 0))
                if img.shape[-1] > 3: 
                    img = img[..., :3]
                if img.dtype != np.uint8:
                    img = (img / img.max() * 255).clip(0, 255).astype(np.uint8)
                
                img = preprocess_input(img.astype('float32'))
                imgs.append(img)
                indices.append(idx)
            except Exception as e:
                continue
            
        if imgs:
            preds = model.predict(np.array(imgs), verbose=0)
            for j, real_idx in enumerate(indices):
                df.at[real_idx, 'img_feature'] = preds[j]
        
        pbar.update(len(imgs))
        
    pbar.close() 
    return df.dropna(subset=['img_feature'])

if __name__ == "__main__":
    BASE_DIR = Path("data/scoring/tiled")
    FOLDERS_TO_PROCESS = ["Gen", "PL", "PLN"]
    
    for folder in FOLDERS_TO_PROCESS:
        folder_path = BASE_DIR / folder
        if not folder_path.exists():
            print(f"Skipping {folder}: Directory not found.")
            continue
            
        print(f"\n{'='*40}\nProcessing Folder: {folder}\n{'='*40}")
        
        # Gather all valid TIF files
        tifs = list(folder_path.glob("*.tif"))
        if not tifs:
            print(f"No TIFs found in {folder}.")
            continue
            
        # Create a dynamic dataframe for the current folder
        df = pd.DataFrame({'chip_path': [str(t) for t in tifs]})
        
        # Run Embedding (Modify model_name and batch_size if needed)
        df_emb = get_embeddings_batch(df, model_name="SatDINO", batch_size=8)
        
        # Save Pickle in the specific folder
        output_pkl = folder_path / "embeddings.pkl"
        df_emb.to_pickle(output_pkl)
        print(f"Successfully saved embeddings to {output_pkl}")