# 🚤 Pleiades Boat Detection (YOLO-OBB)

This project provides a complete pipeline for detecting boats in high-resolution GeoTIFF satellite imagery using **YOLO Oriented Bounding Boxes (OBB)**. It covers everything from tiling large geospatial rasters to stratified dataset splitting and hyperparameter optimization.


## 🚀 Features

### Preprocessing
- **Sensor Degradation**: Implementation of MTF-based blurring, spectral misalignment, and signal-dependent noise modeling.
- **Image Enhancement**: Percentile-based radiometric stretching and Lanczos-4 spatial upsampling.
- **Annotation Mapping**: Projection of GeoJSON OBBs to normalized YOLO formats with automated coordinate mapping.
- **Dataset Splitting**: Stratified partitioning based on class instance counts per image to ensure balanced distributions.

### Training
- **Ultralytics YOLO-OBB**: Integration of Oriented Bounding Box detection for precise localization of non-axis-aligned vessels.
- **Model Initialization**: Support for weight transfer from pretrained checkpoints or training from custom architecture YAMLs.
- **OBB Augmentation**: Native implementation of geometric transforms including mosaic, rotation, and scale jitter.

### Inference & Post-processing
- **Geospatial Mapping**: Pixel-to-world coordinate transformation and standardized GeoJSON output generation.
- **Spatial Filtering**: Modular false-positive suppression via coastline and building mask intersections.
- **Performance Metrics**: Quantitative analysis using OBB-specific mAP, Precision, and Recall.

---

## 📂 Project Structure
```text
pleiades-boat-detection/
├── configs/                    # YAML configuration files
│   ├── preprocessing.yaml      # Configuration for the preprocessing pipeline
│   ├── train.yaml              # Hyperparameters and training settings
│   ├── predict.yaml            # Inference and evaluation parameters
│   ├── postprocessing.yaml     # Spatial filtering and GeoJSON export settings
│   └── yolo26m-obb-p2.yaml     # Model architecture definition
├── data/                       # Dataset storage (ignored by git)
│   ├── raw/                    # Original GeoTIFFs and GeoJSONs
│   └── processed/              # Generated tiles, labels, and splits
├── notebooks/                  # Visualization and analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_model_evaluation_m_320_640.ipynb
├── scripts/                    # CLI entry points
│   ├── preprocessing.py        # Runs the tiling and dataset prep pipeline
│   ├── train.py                # Starts YOLO training
│   ├── predict.py              # Runs inference and calculates metrics
│   ├── grid_search.py          # Hyperparameter optimization
│   └── submit_grid_sequential.py
├── src/                        # Core library
│   └── vessels_detect/
│       ├── manager.py          # Pipeline orchestration
│       ├── preprocessing/      # Tiling, annotations, and dataset splitting
│       │   └── steps/          # Modular preprocessing steps
│       │       ├── annotations.py
│       │       ├── radiometric.py
│       │       ├── spatial.py
│       │       ├── split.py
│       │       └── tiling.py
│       ├── models/             # Model training wrappers
│       │   └── yolo_trainer.py
│       ├── predict/            # Inference, evaluation, and plotting logic
│       │   ├── predictor.py
│       │   ├── evaluation.py
│       │   └── metrics.py
│       ├── postprocessing/     # Geospatial filtering and export
│       │   ├── spatial_filter.py
│       │   ├── geojson_writer.py
│       │   └── steps/          # Modular filtering steps
│       │       ├── buildings.py
│       │       └── coastline.py
│       └── utils/              # CRS handling, config parsing, and helpers
├── weights/                    # Model weights and architecture YAMLs
├── .gitignore
└── README.md
```

---

## 🛠 Installation

---

## 🔄 Workflow Pipeline

The system follows a modular, configuration-driven architecture designed for reproducible geospatial deep learning. The pipeline is divided into three primary phases: Preprocessing, Training, and Inference.

### 1. Data Preprocessing
The preprocessing pipeline transforms raw GeoTIFF imagery and GeoJSON annotations into a format suitable for YOLO-OBB training. It implements a SAHI-optimised global workflow.

*   **Radiometric Correction**: Percentile stretching and gamma correction.
*   **Spatial Resampling**: Lanczos-4 upsampling to enhance feature resolution.
*   **Annotation Mapping**: Conversion of GeoJSON OBBs to normalised YOLO-OBB format.
*   **Stratified Splitting**: Image-level dataset partitioning to prevent spatial leakage.
*   **Optimal Tiling**: Large-scale raster tiling with OBB label projection.

```bash
# Execute the full preprocessing suite
python scripts/preprocessing.py --config configs/preprocessing.yaml
```

### 2. Model Training
The training phase leverages the Ultralytics YOLO framework for Oriented Bounding Box (OBB) detection. It supports multiple initialisation modes (fine-tuning, architectural transfer, or from-scratch).

*   **Configuration**: All hyperparameters, augmentations, and optimiser settings are defined in `configs/train.yaml`.
*   **Execution**: Training progress is monitored via standard logging and optional experiment tracking.

```bash
# Start the training run
python scripts/train.py --config configs/train.yaml
```

### 3. Inference & Evaluation
The prediction pipeline handles inference on unseen geospatial data and calculates rigorous performance metrics.

*   **Inference Mode**: Generates boat detections on raw imagery, applying configured NMS and spatial filtering.
*   **Evaluation Mode**: Matches detections against ground truth to compute Precision, Recall, and mAP@50-95 for OBBs.
*   **Post-processing**: Modular steps to filter false positives using geospatial masks (e.g., coastline, buildings).

```bash
# Run the full inference and evaluation pipeline
python scripts/predict.py --config configs/predict.yaml --mode evaluation
```

---

## 📝 Configuration

## References



