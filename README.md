# 🚤 Pleiades Boat Detection (YOLO-OBB)

## Short Presentation

This project provides an automated, end-to-end deep learning pipeline designed to detect and quantify small-scale artisanal fishing fleets (traditional non-motorized pirogues) along the coasts of Madagascar. Utilizing Very High-Resolution (VHR) satellite imagery (Pléiades Neo at $0.30~m/px$ and Pléiades at $0.50~m/px$), this tool addresses the "statistical invisibility" of artisanal fishing. Built around the Ultralytics YOLO26-OBB (Oriented Bounding Boxes) architecture, the project encompasses everything from raw GeoTIFF radiometric preprocessing and spatial upscaling, to leakage-free dataset stratification, model training, and geospatial post-processing for operational deployment.

## Features

* **Deep Learning for Tiny Objects:** Employs YOLO26 with Oriented Bounding Boxes (OBB) to accurately isolate highly elongated, few-pixel pirogues.
* **Geospatial & Radiometric Precision:** Handles massive raw GeoTIFFs using `rasterio` and `pyproj`. Preserves vital spatial metadata (CRS, Affine transforms) while enabling robust spatial upscaling and radiometric normalization.
* **Global-to-Tile Annotation Engine:** Converts WGS84 GeoJSONs into globally-normalized YOLO labels via `Shapely` minimum rotated rectangles, seamlessly translating them into tile-specific coordinates.
* **Leakage-Free Stratification:** Implements class-aware greedy assignment at the *image* level, strictly separating geographical zones across train/val/test splits to guarantee zero spatial leakage.
* **Automated Spatial Filtering:** Integrates modular geospatial post-processing to automatically suppress false-positive detections by enforcing inclusion within coastline masks and exclusion from building footprints using `geopandas`.

---

## Project Structure

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
│   ├── processed/              # Radiometric, spatial, labels outputs
│   ├── dataset/                # Stratified and tiled dataset ready for training
│   └── eval/                   # Inference inputs, masks, and ground-truth
├── notebooks/                  # Visualisation and analysis
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
├── scripts/                    # CLI entry points
│   ├── preprocessing.py        # Runs the full preprocessing pipeline
│   ├── train.py                # Starts YOLO training
│   ├── predict.py              # Runs inference and calculates metrics
│   └── grid_search/            # Hyperparameter optimisation scripts
├── src/                        # Core library
│   └── vessels_detect/
│       ├── manager.py          # Pipeline orchestration
│       ├── preprocessing/      # Stage 1-6 preprocessing steps
│       ├── degradation/        # Physics-based & stochastic degradation
│       ├── models/             # YOLO_trainer
│       ├── predict/            # Predictor, evaluation, and metrics
│       ├── postprocessing/     # Coastline and building spatial filters
│       └── utils/              
├── weights/                    # Pre-trained checkpoints and custom architectures
├── requirements.txt            # Python dependencies
└── README.md

```

---

## 🔄 Workflow Pipeline


### 1. Data Preprocessing (`configs/preprocessing.yaml`)

The preprocessing pipeline transforms raw GeoTIFF imagery and GeoJSON annotations into tiled YOLO-OBB training data in five sequential steps[cite: 9]. This version is highly optimized, utilizing Python's `ProcessPoolExecutor` to run heavy I/O and CPU tasks concurrently across multiple cores[cite: 9].

```text
raw GeoTIFF + GeoJSON
      │
      ▼ Step 1 & 2 — Image Enhancement & Label Conversion (Parallelized)
  uint8 RGB resampled GeoTIFF + YOLO OBB .txt
      │
      ▼ Step 3 — Dataset Split (Sequential)
  dataset/{images,labels}/{train,val,test}/
      │
      ▼ Step 4 — Slicing (Parallelized)
  tiled/{images,labels}/{train,val,test}/{stem}_{x_off}_{y_off}.{tif,txt}
      │
      ▼ Step 5 — Background Filtering (Sequential)
  relocates excess empty-label tiles to meet target_bg_ratio

```

#### Step 1 & 2 — Image Enhancement & Label Conversion (Parallelized)

To maximize CPU utilization, image enhancement and label conversion are bundled together and executed in parallel for each image.

* **Image Enhancement (`image_enhancement.py`):** Applies a global percentile stretch and gamma correction to the raw GeoTIFF, computing statistics from a fast thumbnail so all windows share consistent color rendering. It then rescales the image to a higher resolution using `rasterio`'s `WarpedVRT` while keeping the geospatial transform correct. Both operations stream the image block-by-block to keep RAM use flat.


* **Label Conversion (`label_conversion.py`):** Converts GeoJSON OBB annotations to YOLO OBB `.txt` label files. Coordinates are normalized to the **full enhanced image dimensions** (not a tile) to ensure compatibility with direct YOLO training and SAHI-based inference. This conversion uses `Shapely` to enforce minimum rotated rectangles and `pyproj` for CRS reprojection.



**Radiometric & Spatial Parameters**

| Key | Default | Description / Impact |
| --- | --- | --- |
| `lo_percentile` | `1.0` | Lower clipping percentile. Drops extreme dark anomalies.

 |
| `hi_percentile` | `99.9` | Upper clipping percentile. Drops bright anomalies.

 |
| `gamma` | `0.8` | Gamma exponent. Values < 1.0 brighten shadows.

 |
| `upscale_ratio` | `2` | Scale factor (e.g. `2` = 2× upsampling).

 |
| `interpolation` | `cubic` | Resampling algorithm (`lanczos`, `cubic`, `bilinear`, `nearest`).

 |

**Annotation Parameters**

| Key | Default | Description / Impact |
| --- | --- | --- |
| `min_visible` | `0.10` | Min fraction of OBB area inside the image boundary.

 |
| `min_size_px` | `2.0` | Min OBB side length in pixels; smaller boxes are symmetrically elongated.

 |
| `class_map` | `{0:0, 1:0...}` | GeoJSON `class_id` → YOLO class index remapping.

 |
| `skip_classes` | `[9, 11]` | GeoJSON class IDs to discard entirely (e.g., buoys).

 |

* **Input:** `paths.raw_dir/*.tif` + `paths.raw_dir/*.geojson` → **Output:** `paths.enhanced_dir/*.tif` + `paths.labels_dir/*.txt`


#### Step 3 — Dataset Split (`dataset_split.py`) (Sequential)

This step runs sequentially on the main thread to ensure deterministic sorting and repeatable dataset splits. It distributes processed images and labels into `train`, `val`, and `test` sub-directories using **class-aware greedy assignment**. Images are assigned at the image level (never split across partitions) to prevent spatial leakage.

| Key | Default | Description / Impact |
| --- | --- | --- |
| `train_ratio` | `0.70` | Target fraction for training data.

 |
| `val_ratio` | `0.15` | Target fraction for validation data.

 |
| `test_ratio` | `0.15` | Target fraction for testing data.

 |
| `priority_class_ids` | `[0]` | Class IDs weighted more heavily in deficit scoring to balance rare classes.

 |
| `priority_weight` | `5.0` | Multiplier for priority classes (≥ 1.0).

 |
| `copy` | `true` | `true` = copy files; `false` = move to save disk space.

 |

* **Input:** `paths.enhanced_dir/*.tif` + `paths.labels_dir/*.txt` → **Output:** `paths.dataset_dir/{images,labels}/{train,val,test}/`


#### Step 4 — Slicing (`slicing.py`) (Parallelized)

Slicing is heavily parallelized across the CPU pool to speed up disk writes. This step cuts the split dataset into fixed-size GeoTIFF patches and projects the image-level YOLO OBB labels into tile-relative normalized coordinates. This stage is **radiometric-free**: source dtype, band count, and raw pixel values are carried through verbatim from the enhanced input.

| Key | Default | Description / Impact |
| --- | --- | --- |
| `splits` | `[train, val]` | Which dataset partitions to process.

 |
| `tile_size` | `1536` | Output tile height and width in pixels.

 |
| `overlap` | `0` | Pixel overlap between adjacent tiles.

 |
| `min_visible_frac` | `0.10` | Min OBB visible fraction to keep a label line inside the tile.

 |

* **Input:** `paths.dataset_dir/{images,labels}/{split}/*.{tif,txt}` → **Output:** `paths.tiled_dir/{images,labels}/{split}/{stem}_{x_off}_{y_off}.{tif,txt}`


#### Step 5 — Background Filtering (`background_filtering.py`) (Sequential)

This step runs sequentially because it must compute the background-to-tile ratio across the entire tiled directory for a given split. It caps the fraction of background (empty-label) tiles by relocating the excess into a `moved/` sub-directory.

| Key | Default | Description / Impact |
| --- | --- | --- |
| `splits` | `[train]` | Which partitions to reduce.

 |
| `target_bg_ratio` | `0.15` | Maximum allowed fraction of empty-label tiles in the dataset (e.g., 15%).

 |
| `moved_subdir` | `moved` | Subdirectory where excess background tiles are relocated.

 |

* **Input:** `paths.tiled_dir/{images,labels}/{split}/*.{tif,txt}` → **Output:** Moves excess to `paths.tiled_dir/moved/{images,labels}/`


---

### 2. Model Training (`configs/train.yaml`)

The training phase utilizes the `ultralytics` framework for OBB detection. It manages transfer learning and data augmentation.

#### Model & Training Parameters

| Key | Default | Description / Impact |
| --- | --- | --- |
| `model.weights` | `weights/yolo26m-obb.pt` | Architecture/Weights to load. `.pt` triggers fine-tuning. `.yaml` builds a custom architecture (e.g., P2 head). |
| `training.epochs` | `100` | Maximum number of training epochs. |
| `training.imgsz` | `2048` | Target tensor size for the network. |
| `training.batch_size` | `2` | Number of tiles per batch. Kept low due to heavy VHR memory constraints. |
| `training.patience` | `20` | Early stopping trigger. Stops if no val metrics improve for 20 epochs. |

#### Augmentations (Geometric & Photometric)

| Key | Default | Description / Impact |
| --- | --- | --- |
| `augmentation.hsv_h` | `0.015` | Hue shift. Simulates different water colors/turbidity. |
| `augmentation.mosaic` | `0.7` | Probability of combining 4 images into 1. Enhances contextual learning but alters object scale. |
| `augmentation.degrees` | `180.0` | Rotation range (±180). Crucial for OBB since pirogues orient in all directions. |
| `augmentation.scale` | `0.30` | Scale jitter. Helps the network generalize to different pirogue lengths. |

* **Input:** `data/dataset.yaml` (Pointing to Stage 5/6 outputs) → **Output:** `runs/boat_obb/weights/best.pt`

---

### 3. Inference & Post-processing (`configs/predict.yaml` & `configs/postprocessing.yaml`)

The prediction pipeline generates detections, merges tiles, suppresses false positives using geospatial logic (`geopandas`), and evaluates metrics.

#### Prediction & Evaluation (`predict.yaml`)

| Key | Default | Description / Impact |
| --- | --- | --- |
| `pipeline.mode` | `evaluation` | `inference` (generate outputs) vs `evaluation` (generate outputs + match GT to calculate mAP). |
| `model.weights` | `weights/50_x4_best.pt` | Path to the trained checkpoint to load for inference. |
| `prediction.conf` | `0.20` | Confidence threshold. Detections below this probability are dropped. |
| `prediction.global_nms_iou` | `0.20` | IoU threshold for Global NMS to remove duplicates created on overlapping tile boundaries. |
| `evaluation.iou_threshold` | `0.25` | PASCAL VOC threshold. An overlap ≥ 25% with ground truth is required to count as a True Positive. |

#### Spatial Filters (`postprocessing.yaml`)

Executes purely geometric operations to prune false positives found on land.

| Key | Default | Description / Impact |
| --- | --- | --- |
| `coastline_filter.min_area_fraction` | `0.80` | Requires ≥80% of a predicted OBB to sit inside the water mask. |
| `buildings_filter.max_overlap_fraction` | `0.01` | Drops a prediction if ≥1% of its area overlaps a building polygon. |

* **Input:** Raw Test GeoTIFFs + Checkpoint → **Output:** `predictions/raw/` → **Filtered Output:** `predictions/postprocessed/` + `results.csv`

---

## 💻 List of Commands to Run the Code

The system uses configuration-driven CLI entry points. You do not need to modify Python code to toggle stages or tweak parameters.

```bash
# 1. Run the full preprocessing pipeline (Stages 1 through 6)
python scripts/preprocessing.py --config configs/preprocessing.yaml

# 1b. Rerun only specific stages (e.g., skip to tiling)
python scripts/preprocessing.py --config configs/preprocessing.yaml --stages tiling background_reduction

# 2. Train the YOLO-OBB model
python scripts/train.py --config configs/train.yaml

# 3. Run prediction in Inference mode (Outputs raw & postprocessed GeoJSONs)
python scripts/predict.py --config configs/predict.yaml --mode inference

# 4. Run prediction in Evaluation mode (Matches GT, outputs mAP50 metrics & TP/FP/FN labeled GeoJSONs)
python scripts/predict.py --config configs/predict.yaml --mode evaluation

```

---

## 📚 References & Dependencies

**Core Libraries Used in Code (`requirements.txt`):**

* `rasterio` (1.5.0): Windowed I/O and WarpedVRT resampling.
* `geopandas` (0.14.3) & `Shapely` (2.1.2): Post-processing spatial intersections and OBB `minimum_rotated_rectangle` logic.
* `pyproj` (3.7.2): Coordinate Reference System transformations.
* `ultralytics` (8.4.1): Core YOLO backend for training and inference.
* `affine` (2.4.0), `numpy` (2.4.4), `pandas` (3.0.2), `PyYAML` (6.0.3).

**Academic & Technical References:**

1. Airbus Defence and Space: Pléiades Neo User Guide. Airbus DS (2021).
2. Basurto, X., et al.: Illuminating the multidimensional contributions of small-scale fisheries. *Nature* 637, 875-884 (2025).
3. Cheng, G., et al.: Towards Large-Scale Small Object Detection: Survey and Benchmarks. *IEEE TPAMI* (2023).
4. Ding, J., et al.: Learning RoI Transformer for Detecting Oriented Objects in Aerial Images. arXiv (2018).
5. Jocher, G., Qiu, J., Chaurasia, A.: Ultralytics YOLO (2023). [GitHub](https://github.com/ultralytics/ultralytics)
6. Luo, W., et al.: Understanding the Effective Receptive Field in Deep Convolutional Neural Networks. arXiv (2017).
7. Zucchetta, M., et al.: Satellite-based monitoring of small boat for environmental studies: A systematic review. *JMSE* (2025).