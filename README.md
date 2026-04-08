# 🚤 Pleiades Boat Detection (YOLO-OBB)

This project provides a complete pipeline for detecting boats in high-resolution GeoTIFF satellite imagery using **YOLO Oriented Bounding Boxes (OBB)**. It covers everything from raw GeoTIFF preprocessing to stratified dataset splitting, tiling, training, and evaluation.

---

## 🚀 Features

### Preprocessing
- **Radiometric Normalisation**: Global percentile stretching with gamma correction, computed from a thumbnail for consistent colour rendering across the full image. Output: uint8 RGB GeoTIFF.
- **Spatial Resampling**: Configurable upsampling via rasterio `WarpedVRT` (Lanczos, Cubic, Bilinear, Nearest). Block-wise streaming preserves geospatial metadata (CRS, Affine transform).
- **Annotation Conversion**: GeoJSON OBB → YOLO OBB label files normalised to the **global** processed image dimensions. Includes visibility filtering, minimum-side enforcement for degenerate boxes, and class remapping.
- **Stratified Splitting**: Image-level train/val/test partitioning with class-aware greedy assignment to balance rare-class representation and prevent spatial leakage.
- **Raw GeoTIFF Tiling**: Overlap-aware tiling of the split dataset with YOLO OBB label projection into tile space. Source dtype, band count, CRS, and raw pixel values are preserved verbatim. Uniform tiles (all pixels identical) are discarded.

### Training
- **Ultralytics YOLO-OBB**: Oriented Bounding Box detection for precise localisation of non-axis-aligned vessels.
- **Model Initialisation**: Weight transfer from pretrained checkpoints or training from custom architecture YAMLs.
- **OBB Augmentation**: Geometric transforms including mosaic, rotation, and scale jitter.

### Inference & Post-processing
- **Geospatial Output**: Pixel-to-world coordinate transformation with standardised GeoJSON export.
- **Spatial Filtering**: Modular false-positive suppression via coastline and building mask intersections.
- **Performance Metrics**: OBB-specific mAP, Precision, and Recall with confidence-sorted greedy IoU matching (PASCAL VOC protocol).

---

## 📂 Project Structure

```text
pleiades-boat-detection/
├── configs/                    # YAML configuration files
│   ├── preprocessing.yaml      # Full preprocessing pipeline configuration
│   ├── train.yaml              # Hyperparameters and training settings
│   ├── predict.yaml            # Inference and evaluation parameters
│   ├── postprocessing.yaml     # Spatial filtering and GeoJSON export settings
│   └── yolo26m-obb-p2.yaml     # Model architecture definition
├── data/                       # Dataset storage (ignored by git)
│   ├── raw/                    # Original GeoTIFFs and GeoJSON annotations
│   └── processed/              # Radiometric, spatial, labels, dataset, tiled outputs
├── notebooks/                  # Visualisation and analysis
│   ├── 01_data_exploration.ipynb
│   └── 02_model_evaluation_m_320_640.ipynb
├── scripts/                    # CLI entry points
│   ├── preprocessing.py        # Runs the full preprocessing pipeline
│   ├── train.py                # Starts YOLO training
│   ├── predict.py              # Runs inference and calculates metrics
│   ├── grid_search.py          # Hyperparameter optimisation
│   └── submit_grid_sequential.py
├── src/                        # Core library
│   └── vessels_detect/
│       ├── manager.py          # Pipeline orchestration
│       ├── preprocessing/
│       │   ├── manager.py      # Registry + PreprocessingManager
│       │   └── steps/
│       │       ├── base.py         # Abstract BaseStep
│       │       ├── radiometric.py  # Stage 1 — percentile stretch + gamma
│       │       ├── spatial.py      # Stage 2 — WarpedVRT resampling
│       │       ├── annotations.py  # Stage 3 — GeoJSON OBB → YOLO OBB
│       │       ├── split.py        # Stage 4 — image-level stratified split
│       │       └── tiling.py       # Stage 5 — raw GeoTIFF tiling
│       ├── models/
│       │   └── yolo_trainer.py
│       ├── predict/
│       │   ├── predictor.py
│       │   ├── evaluation.py
│       │   └── metrics.py
│       ├── postprocessing/
│       │   ├── spatial_filter.py
│       │   ├── geojson_writer.py
│       │   └── steps/
│       │       ├── buildings.py
│       │       └── coastline.py
│       └── utils/
├── weights/                    # Model weights and architecture YAMLs
├── .gitignore
└── README.md
```

---

## 🛠 Installation

---

## 🔄 Workflow Pipeline

The system follows a modular, configuration-driven registry architecture. Each stage is a `BaseStep` subclass registered in `STEP_REGISTRY`; adding a new stage requires no changes outside its own module and the registry entry.

### 1. Data Preprocessing

The preprocessing pipeline transforms raw GeoTIFF imagery and GeoJSON annotations into tiled YOLO-OBB training data in five sequential stages.

```
raw GeoTIFF + GeoJSON
      │
      ▼ Stage 1 — radiometric
  uint8 RGB GeoTIFF  (global percentile stretch + gamma)
      │
      ▼ Stage 2 — spatial
  resampled GeoTIFF  (WarpedVRT, configurable scale + interpolation)
      │
      ▼ Stage 3 — annotations
  YOLO OBB .txt  (one file per image, coords normalised to full image)
      │
      ▼ Stage 4 — split
  dataset/{images,labels}/{train,val,test}/
      │
      ▼ Stage 5 — tiling
  tiled/{images,labels}/{train,val,test}/{stem}_{x_off}_{y_off}.{tif,txt}
```

#### Stage 1 — Radiometric Normalisation (`radiometric.py`)

Applies a **global** percentile stretch followed by gamma correction to each raw GeoTIFF. Statistics are computed from a fast bilinear thumbnail (longest edge: 1024 px) so that all windows share a consistent colour rendering — eliminating per-tile contrast drift that confuses feature extractors. The full-resolution image is written band-by-band via windowed I/O.


Key config (`configs/preprocessing.yaml → radiometric`):

| Key | Default | Description |
|---|---|---|
| `lo_percentile` | `1.0` | Lower clipping percentile |
| `hi_percentile` | `99.9` | Upper clipping percentile |
| `gamma` | `0.8` | Gamma exponent (< 1.0 brightens shadows) |
| `bands` | `null` | 1-based source band indices for (R,G,B); `null` = auto |
| `compress` | `lzw` | GeoTIFF compression codec |

Input: `paths.raw_dir/*.tif` → Output: `paths.radiometric_dir/*.tif`

#### Stage 2 — Spatial Resampling (`spatial.py`)

Rescales each GeoTIFF using rasterio's `WarpedVRT` API, which correctly handles global resampling while preserving the CRS and updating the Affine transform so that geographic extent is identical to the input.

Key config (`configs/preprocessing.yaml → spatial`):

| Key | Default | Description |
|---|---|---|
| `upscale_ratio` | `1.0` | Scale factor (e.g. `2` = 2× upsampling) |
| `interpolation` | `lanczos` | Resampling algorithm: `lanczos`, `cubic`, `bilinear`, `nearest` |
| `window_size` | `512` | Block size for streaming I/O |
| `compress` | `lzw` | GeoTIFF compression codec |

Input: `paths.radiometric_dir/*.tif` → Output: `paths.spatial_dir/*.tif`

#### Stage 3 — Annotation Conversion (`annotations.py`)

Converts GeoJSON OBB annotations to YOLO OBB `.txt` label files. Coordinates are normalised to the **full processed image dimensions** — not a tile — so that label files are valid for both direct YOLO training and SAHI-based inference.

Coordinate flow per annotation:
```
GeoJSON exterior ring  (WGS 84 / EPSG:4326)
    ↓  minimum_rotated_rectangle  (Shapely)
    ↓  reproject to image CRS  (pyproj Transformer)
    ↓  apply inverse image Affine  →  pixel (col, row)
    ↓  enforce minimum side length  (symmetric elongation from centroid)
    ↓  normalise by (image_width, image_height)  →  [0, 1]
    ↓  write YOLO OBB line:  class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Key config (`configs/preprocessing.yaml → annotations`):

| Key | Default | Description |
|---|---|---|
| `min_visible` | `0.10` | Min fraction of OBB area inside the image boundary |
| `min_size_px` | `2.0` | Min OBB side length in pixels; smaller boxes are elongated |
| `class_map` | `{0:0, …}` | GeoJSON `class_id` → YOLO class index remapping |
| `skip_classes` | `[9, 11]` | GeoJSON class IDs to discard entirely |

Input: `paths.spatial_dir/*.tif` + `paths.raw_dir/*.geojson` → Output: `paths.labels_dir/*.txt`

#### Stage 4 — Dataset Split (`split.py`)

Distributes processed images and labels into `train`, `val`, and `test` sub-directories using **class-aware greedy assignment**. Images are assigned at the image level (never split across partitions) to prevent spatial leakage. Images with the most priority-class annotations are assigned first.

Scoring: each candidate split is scored by how much adding the image reduces its per-class deficit (weighted by `priority_weight` for priority classes). Ties are broken randomly with a fixed seed.

Key config (`configs/preprocessing.yaml → split`):

| Key | Default | Description |
|---|---|---|
| `train_ratio` | `0.70` | Target fraction for training |
| `val_ratio` | `0.15` | Target fraction for validation |
| `test_ratio` | `0.15` | Target fraction for testing |
| `priority_class_ids` | `[0]` | Class IDs weighted more heavily in deficit scoring |
| `priority_weight` | `5.0` | Multiplier for priority classes (≥ 1.0) |
| `random_seed` | `42` | Seed for reproducible tie-breaking |
| `copy` | `false` | `true` = copy files (keep originals); `false` = move |

Input: `paths.spatial_dir/*.tif` + `paths.labels_dir/*.txt` → Output: `paths.dataset_dir/{images,labels}/{train,val,test}/`

#### Stage 5 — Tiling (`tiling.py`)

Tiles the split dataset into fixed-size GeoTIFF patches and projects YOLO OBB labels from image space into tile space. This stage is **radiometric-free**: source dtype, band count, CRS, and raw pixel values are preserved verbatim. Each tile embeds its own CRS and Affine transform (anchored at its top-left corner) plus provenance TIFF tags.

Uniform tiles (global `min == max` across all bands) are discarded — these are nodata slabs, pure-water expanses, or padding-only edge tiles with no information content.

Label projection: for each tile, all source annotations are evaluated against the tile polygon in pixel space. Annotations whose visible area (`intersection / OBB area`) is below `min_visible_frac` are discarded. Surviving annotations are re-expressed in tile-relative normalised coordinates; corners outside `[0, 1]` are intentional and valid for partial boxes at tile edges.

Key config (`configs/preprocessing.yaml → tiling`):

| Key | Default | Description |
|---|---|---|
| `splits` | `[train, val, test]` | Which splits to tile |
| `tile_size` | `640` | Output tile height and width in pixels |
| `overlap` | `64` | Pixel overlap between adjacent tiles; stride = `tile_size - overlap` |
| `compress` | `lzw` | GeoTIFF compression codec |
| `min_visible_frac` | `0.10` | Min OBB visible fraction to keep a label line |
| `images_subdir` | `images` | Subdirectory name for tile GeoTIFFs inside `tiled_dir` |
| `labels_subdir` | `labels` | Subdirectory name for tile label files inside `tiled_dir` |

Input: `paths.dataset_dir/{images,labels}/{split}/*.{tif,txt}` → Output: `paths.tiled_dir/{images_subdir,labels_subdir}/{split}/{stem}_{x_off}_{y_off}.{tif,txt}`

### 2. Training
[TODO] 

### 3. Inference & Post-processing
[TODO] 


#### Running the pipeline

```bash
# Full pipeline
python scripts/preprocessing.py --config configs/preprocessing.yaml

# Partial rerun — specific stages only (skips enabled flags)
python scripts/preprocessing.py --config configs/preprocessing.yaml \
    --stages tiling
```

Individual stages can also be disabled without touching the code by setting `enabled: false` in the `stages` list in `configs/preprocessing.yaml`.

---

### 2. Model Training

The training phase uses the Ultralytics YOLO framework for OBB detection.

- **Configuration**: All hyperparameters, augmentations, and optimiser settings are defined in `configs/train.yaml`.
- **Initialisation modes**: fine-tuning from a pretrained checkpoint, architectural transfer, or from-scratch training via a custom YAML.

```bash
python scripts/train.py --config configs/train.yaml
```

---

### 3. Inference & Evaluation

The prediction pipeline handles inference on geospatial data and computes rigorous OBB metrics.

- **Inference mode**: generates detections on raw imagery with configured NMS and spatial filtering.
- **Evaluation mode**: matches detections against ground truth using confidence-sorted greedy IoU matching (PASCAL VOC protocol) and computes per-class and global Precision, Recall, F1, and mAP@50.
- **Post-processing**: modular false-positive suppression via coastline and building masks.

```bash
python scripts/predict.py --config configs/predict.yaml --mode evaluation
```

---

## 📝 Configuration

All pipeline behaviour is driven by YAML files in `configs/`. The preprocessing pipeline reads a single file (`preprocessing.yaml`) that controls paths, enabled stages, and all per-stage hyperparameters. No code changes are needed to add, skip, or reorder stages.

[TODO] - training + Inference & Post-processing
---

## References

> **Ultralytics YOLO26**
> Jocher, G., & Qiu, J. (2026). *Ultralytics YOLO26* (Version 26.0.0) [Computer software]. Available at [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (License: AGPL-3.0)