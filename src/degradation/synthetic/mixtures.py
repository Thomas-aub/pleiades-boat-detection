"""
mixtures.py — Tile collection and stochastic degradation-mixture assignment.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

logger = logging.getLogger(__name__)

_IMAGE_GLOBS = ("*.tif", "*.tiff")

# ============================================================================
# Tools — pure functions
# ============================================================================

def _load_annotations(labels_dir: Path, image_stem: str) -> List[List[tuple]]:
    p = labels_dir / f"{image_stem}.geojson"
    if not p.exists():
        return []
    return [f["geometry"]["coordinates"][0]
            for f in json.loads(p.read_text(encoding="utf-8"))["features"]]

def _tile_has_annotation(annots: List[List[tuple]], tile: dict, tf: Transformer) -> bool:
    for ring in annots:
        cx = sum(c[0] for c in ring) / len(ring)
        cy = sum(c[1] for c in ring) / len(ring)
        x, y = tf.transform(cx, cy)
        if tile["min_x"] <= x <= tile["max_x"] and tile["min_y"] <= y <= tile["max_y"]:
            return True
    return False

def _bg_keep(n_pos: int, ratio: float) -> int:
    return int(n_pos * ratio / (1.0 - ratio))

def _tile_windows(tif_path: Path, tile_size: int) -> List[dict]:
    tiles = []
    with rasterio.open(tif_path) as ds:
        if ds.crs is None:
            return tiles
        t = ds.transform
        gsd_x, gsd_y = abs(t.a), abs(t.e)
        for row in range(0, ds.height, tile_size):
            for col in range(0, ds.width, tile_size):
                c1, r1        = min(col + tile_size, ds.width), min(row + tile_size, ds.height)


                # Drop edge tiles — partial windows produce variable output sizes after downsampling
                if (c1 - col) < tile_size or (r1 - row) < tile_size:
                    continue

                left, top     = t * (col, row)
                right, bottom = t * (c1, r1)
                tiles.append({
                    "_tif_path":       tif_path,
                    "_window":         Window(col, row, c1 - col, r1 - row),
                    "image_id":        tif_path.name,
                    "tile_id":         f"{row}_{col}",
                    "min_x":           min(left, right),
                    "max_x":           max(left, right),
                    "min_y":           min(top, bottom),
                    "max_y":           max(top, bottom),
                    "tile_size_px":    tile_size,
                    "length_x_meters": tile_size * gsd_x,
                    "length_y_meters": tile_size * gsd_y,
                    "native_crs":      ds.crs.to_string(),
                    "GSD_input":       round((gsd_x + gsd_y) / 2.0, 4),
                })
    return tiles

def _sample_mixture(cfg: dict, i: int) -> dict:
    """Sample one randomised stochastic degradation configuration."""
    steps = ["B", "D", "N"]
    random.shuffle(steps)
    
    # Uniform distributions based on Real-ESRGAN paradigms adapted for remote sensing
    blur_types = ["iso", "aniso", "generalized", "sinc"]
    noise_types = ["gaussian", "poisson"]
    
    return {
        "Transform_id": i,
        "step_order":   steps,
        "blur_type":    random.choice(blur_types),
        "noise_type":   random.choice(noise_types),
        "snr_db":       round(random.uniform(*cfg["snr_limits"]), 2),
        "psf_size":     round(random.uniform(*cfg["psf_limits"]), 3),
        "gsd_output":   round(random.uniform(*cfg["gsd_limits"]), 4) # Continuous downsampling
    }

def _build_record(tile: dict, mix: dict) -> dict:
    return {
        **{k: v for k, v in tile.items() if not k.startswith("_")},
        "Transform_id": mix["Transform_id"],
        "Order":        "".join(mix["step_order"]),
        "Blur_Type":    mix["blur_type"],
        "Noise_Type":   mix["noise_type"],
        "SNR_dB":       mix["snr_db"],
        "PSF":          mix["psf_size"],
        "GSD_output":   mix["gsd_output"]
    }

# ============================================================================
# Workers
# ============================================================================

def generate_pool(cfg: dict) -> List[dict]:
    random.seed(cfg["random_seed"])
    pool = [_sample_mixture(cfg, i) for i in range(cfg["n_mixtures"])]
    logger.info("Generated %d stochastic mixtures.", cfg["n_mixtures"])
    return pool

def collect_tiles(cfg: dict) -> List[dict]:
    input_dir = Path(cfg["input_directory"])
    tif_files = sorted(p for g in _IMAGE_GLOBS for p in input_dir.glob(g))
    tiles = []
    for tif in tif_files:
        tiles.extend(_tile_windows(tif, cfg["tile_size"]))
    logger.info("Collected %d tiles from %d file(s).", len(tiles), len(tif_files))
    return tiles

def filter_nodata(tiles: List[dict], cfg: dict) -> List[dict]:
    if cfg["max_nodata_pct"] >= 1.0:
        return tiles
    by_file: Dict[Path, List[dict]] = {}
    for t in tiles:
        by_file.setdefault(t["_tif_path"], []).append(t)
    kept = []
    for tif_path, group in by_file.items():
        with rasterio.open(tif_path) as ds:
            nd = ds.nodata or 0
            for tile in group:
                raw = ds.read(window=tile["_window"])
                check_raw = raw[:3] if raw.shape[0] >= 3 else raw
                nd_pct = np.all(check_raw == nd, axis=0).sum() / (raw.shape[1] * raw.shape[2])
                if nd_pct > cfg["max_nodata_pct"] or (check_raw.max() == check_raw.min()):
                    continue
                kept.append(tile)
    return kept

def reduce_background(tiles: List[dict], cfg: dict) -> List[dict]:
    if not cfg.get("labels_dir"): return tiles
    # Logic unchanged from original...
    labels_dir = Path(cfg["labels_dir"])
    by_image: Dict[str, List[dict]] = {}
    for tile in tiles: by_image.setdefault(tile["image_id"], []).append(tile)
    pos, bg = [], []
    for image_id, img_tiles in by_image.items():
        annots = _load_annotations(labels_dir, Path(image_id).stem)
        tf = Transformer.from_crs("EPSG:4326", img_tiles[0]["native_crs"], always_xy=True)
        for tile in img_tiles:
            (pos if _tile_has_annotation(annots, tile, tf) else bg).append(tile)
    if not pos: return tiles
    keep = _bg_keep(len(pos), cfg["target_bg_ratio"])
    random.seed(cfg["random_seed"])
    random.shuffle(bg)
    return pos + bg[:keep]

def assign_mixtures(tiles: List[dict], pool: List[dict], cfg: dict) -> List[dict]:
    records = [
        _build_record(tile, mix)
        for tile in tiles
        for mix in random.sample(pool, k=cfg["n_mixtures_per_tile"])
    ]
    return records

def export_csv(records: List[dict], cfg: dict) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df.to_csv(cfg["output_csv"], index=False)
    return df

def run(cfg: dict) -> List[dict]:
    pool    = generate_pool(cfg)
    tiles   = collect_tiles(cfg)
    tiles   = filter_nodata(tiles, cfg)
    tiles   = reduce_background(tiles, cfg)
    records = assign_mixtures(tiles, pool, cfg)
    export_csv(records, cfg)
    return records

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = {
        "input_directory":     "./data/raw",
        "output_csv":          "tile_mixture_assignments.csv",
        "gsd_limits":          [0.45, 0.55], # Replaces fixed GSD_output for continuous resize variation
        "tile_size":           427,
        "psf_limits":          [0.48, 0.65], 
        "snr_limits":          [40, 44],
        "n_mixtures":          50,
        "n_mixtures_per_tile": 3,
        "max_nodata_pct":      0.50,
        "labels_dir":          None,
        "target_bg_ratio":     0.15,
        "random_seed":         42,
    }
    run(cfg)