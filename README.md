# High-Resolution GeoSpatial Boat Annotation & YOLO OBB Training

This project provides a complete pipeline for detecting boats in high-resolution GeoTIFF satellite imagery using **YOLO Oriented Bounding Boxes (OBB)**. It covers everything from tiling large geospatial rasters to stratified dataset splitting and hyperparameter optimization.

## 🚀 Features
- **Geospatial Preprocessing**: Tiles large `.tif` files into standard 1024x1024 PNGs with global contrast stretching.
- **OBB Label Conversion**: Converts GeoJSON boat annotations (WGS84) to YOLO OBB format, handling coordinate transformations and class mapping.
- **Stratified Splitting**: Ensures balanced class distribution across train, validation, and test sets.
- **Background Downsampling**: Controls the ratio of empty (water/land) tiles to improve model focus.
- **Hyperparameter Optimization (HPO)**: Uses Optuna, Hydronaut, and MLflow for automated tuning and experiment tracking.

---

## 📂 Project Structure
```text
.
├── scripts/
│   ├── 01_preprocessing_images.py  # Tile large GeoTIFFs to PNG
│   ├── 02_geojson_to_txt.py        # GeoJSON OBB -> YOLO OBB txt
│   ├── 03_split_dataset.py         # Stratified train/val/test split
│   ├── 04_downsample_background.py # Reduce empty background tiles
│   ├── 05_model.py                 # Standard YOLO OBB training
│   └── 06_model_optuna.py          # HPO with Optuna + MLflow
├── data/
│   ├── raw/                        # Original .tif and .geojson files
│   ├── processed/                  # Generated tiles and labels
│   └── dataset.yaml                # YOLO configuration
├── conf/
│   └── config.yaml                 # HPO configuration
└── requirements.txt
```

---

## 🛠 Installation
Ensure you have Python 3.9+ and run:
```bash
pip install -r requirements.txt
```
*Note: Some geospatial libraries like `rasterio` may require system-level dependencies (GDAL).*

---

## 🔄 Workflow Pipeline

### 1. Preprocessing Images
Tiles your large `.tif` files and applies global contrast stretching to prevent color drift between tiles.
```bash
python scripts/01_preprocessing_images.py
```

### 2. Label Conversion
Converts your GeoJSON boat annotations into YOLO OBB format (`class_id x1 y1 x2 y2 x3 y3 x4 y4`).
```bash
python scripts/02_geojson_to_txt.py
```

### 3. Dataset Splitting
Performs a stratified split based on the presence of specific boat classes in each tile.
```bash
python scripts/03_split_dataset.py
```

### 4. Downsampling Background
Reduces the number of empty water tiles to a target ratio (default 10%) to prevent model bias.
```bash
python scripts/04_downsample_background.py
```

### 5. Training the Model
Train a standard YOLO OBB model (nano, small, medium, etc.).
```bash
python scripts/05_model.py
```

### 6. Hyperparameter Optimization
Run automated tuning for the best detection results.
```bash
python scripts/06_model_optuna.py
```

---

## 📊 Monitoring & Results
- **MLflow**: Track all HPO trials.
  ```bash
  mlflow ui --backend-store-uri runs/mlflow.db
  ```
- **Optuna Dashboard**: View optimization progress live.
  ```bash
  optuna-dashboard sqlite:///runs/optuna/boat_obb_study/optuna_study.db
  ```
- **YOLO Logs**: Standard training metrics are stored in `runs/obb/`.

---

## 📝 Configuration
- **Tile Size**: Default is `1024x1024` with `64px` overlap.
- **Classes**: Mapping is defined in `02_geojson_to_txt.py`.
- **Optimization**: The search space and trials can be adjusted in `06_model_optuna.py`.
