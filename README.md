# High-Resolution GeoSpatial Boat Annotation & YOLO OBB Training

This project provides a complete pipeline for detecting boats in high-resolution GeoTIFF satellite imagery using **YOLO Oriented Bounding Boxes (OBB)**. It covers everything from tiling large geospatial rasters to stratified dataset splitting and hyperparameter optimization.

### TODO
- [ ]  Move "5 - Dataset statistics" & "6 - OBB geometric statistics" from  "notebooks/02_model_evaluation.ipynb" to "notebooks/01_data_exploration.ipynb"
- [ ]  Make sure inference is working
- [ ]  .gitignore
- [ ]  This README.md

## 🚀 Features

---

## 📂 Project Structure
```text
pleiades_boat_detection/
├── configs/                    # YAML configuration files (replaces hardcoded globals)
│   ├── data_prep.yaml
│   ├── train.yaml
│   └── inference.yaml
├── data/                       # Ignored by git (.gitignore)
│   ├── raw/                    # Original GeoTIFFs and GeoJSONs
│   └── processed/              # Tiles, labels, metadata, splits
├── notebooks/                  # Only for visualization and exploration
│   ├── 01_data_exploration.ipynb
│   └── 02_model_evaluation.ipynb  # Replaces model_results_640_bi.ipynb
├── scripts/                    # Executable entry points (thin wrappers)
│   ├── prep_data.py            # Replaces 00 to 05
│   ├── train.py                # Replaces 06
│   └── predict.py              # New inference script
├── src/                        # The core Python library
│   └── pleiades_detect/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── tiler.py        # Logic from 01_preprocessing_images.py
│       │   ├── annotations.py  # Logic from 02_geojson_to_txt.py
│       │   ├── split.py        # Logic from 03_split_dataset.py & 04_downsample
│       │   └── transforms.py   # Logic from 05_upsample.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── yolo_trainer.py # Logic from 06_model.py
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predictor.py    # Runs YOLO inference on tiles
│       │   ├── postprocess.py  # NMS, tile-stitching
│       │   └── geospatial.py   # Pixel-to-GeoJSON conversion mapping
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py      # Metric logic extracted from your Notebook
│       └── utils/
│           ├── __init__.py
│           ├── config.py       # YAML parsing (e.g., using OmegaConf/Hydra)
│           └── logger.py       # Standard logging (replaces print statements)
├── .gitignore
├── pyproject.toml              # Modern Python dependency/package manager
└── README.md
```

---

## 🛠 Installation

---

## 🔄 Workflow Pipeline


---


## 📝 Configuration



