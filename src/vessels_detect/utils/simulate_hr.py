# pip install git+https://github.com/JulioContrerasH/satharmony

"""Pléiades Neo → Pléiades (PHR 1A/1B) sensor emulation.

Transforms Pléiades Neo MS imagery to the radiometric and spatial
characteristics of the Pléiades (PHR 1A/1B) MS product.

--------------------------------------------------------------------
Pléiades (PHR 1A/1B):
* GSD: 0.5 m (Pansharpened), 2.0 m (Raw MS)
* Spectral bands: B, G, R, NIR (4 bands)
* Quantization: 12-bit
* MTF @ Nyquist: ~0.10 (PAN) to ~0.15 (MS) -> Requires a target Gaussian PSF sigma of ~0.6 to 0.75 pixels.
* SNR: ~200 (typical) -> 20 * log10(200) ≈ 46.0 dB
* Dynamic range: 0–4095 DN

Pléiades Neo (3/4):
* GSD: 0.3 m (Pansharpened), 1.2 m (4-band MS), 0.9 m (6-band MS)
* Spectral bands: 6 MS (Deep Blue, B, G, Y, R, RE, NIR) + PAN
* Quantization: 12-bit
* MTF @ Nyquist: ~0.15 (PAN), ~0.20 (MS) -> Sharper than PHR.
* SNR: ~250 (higher sensitivity) -> 20 * log10(250) ≈ 48.0 dB
--------------------------------------------------------------------
"""

import gc
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
from satharmony import MSSEmulator, PipelineConfig

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: 'tqdm' is required. Run: pip install tqdm")


# ---------------------------------------------------------------------------
# GSD constants
# ---------------------------------------------------------------------------

_PHR_MS_GSD = 0.5
_NEO_MS_GSD = 0.3
_GSD_RATIO  = _PHR_MS_GSD / _NEO_MS_GSD   # 5/3 ≈ 1.667 — output has fewer pixels

# Neo imagery is 12-bit; normalise against the physical sensor ceiling instead
# of per-tile min/max so radiometry stays consistent across tile boundaries.
_NEO_DN_MAX = 4095.0

# ---------------------------------------------------------------------------
# Tiling constants  (tune these to fit your available RAM)
# ---------------------------------------------------------------------------

TILE_SIZE    = 2048   # input pixels per tile side
TILE_OVERLAP = 0    # overlap pixels (absorbs PSF/blur edge artefacts)


# ---------------------------------------------------------------------------
# Tiling helpers  (ported from degrade_pipeline.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileConfig:
    tile_size: int
    overlap:   int

    @property
    def stride(self) -> int:
        return max(1, self.tile_size - self.overlap)


@dataclass(frozen=True)
class Tile:
    read_window:  Window   # what to read from the source (includes overlap)
    write_window: Window   # the central core to commit to the output


def iter_tiles(width: int, height: int, cfg: TileConfig):
    """Yield Tiles covering the full raster with symmetric overlap padding.

    Only the central ``write_window`` is committed to disk; the overlap guards
    against edge artefacts produced by the emulator's spatial operations.
    """
    for y0 in range(0, height, cfg.stride):
        for x0 in range(0, width, cfg.stride):
            core_w = min(cfg.stride, width  - x0)
            core_h = min(cfg.stride, height - y0)
            pad_left   = min(x0,               cfg.overlap // 2)
            pad_top    = min(y0,               cfg.overlap // 2)
            pad_right  = min(width  - (x0 + core_w), cfg.overlap - pad_left)
            pad_bottom = min(height - (y0 + core_h), cfg.overlap - pad_top)

            rx, ry = x0 - pad_left, y0 - pad_top
            rw = pad_left + core_w + pad_right
            rh = pad_top  + core_h + pad_bottom

            yield Tile(
                read_window=Window(rx, ry, rw, rh),
                write_window=Window(x0, y0, core_w, core_h),
            )


def _output_window(tile: Tile, scale: float, out_w: int, out_h: int) -> Window:
    """Map a source write-window to the destination pixel grid.

    ``scale`` is the spatial downscale factor (GSD_out / GSD_in > 1 here),
    so the output window is smaller than the source window.
    Clamps to image bounds to prevent off-by-one overflows on the last tile.
    """
    w  = tile.write_window
    ox = round(w.col_off / scale)
    oy = round(w.row_off  / scale)
    ow = min(round((w.col_off + w.width)  / scale) - ox, out_w - ox)
    oh = min(round((w.row_off + w.height) / scale) - oy, out_h - oy)
    return Window(ox, oy, max(1, ow), max(1, oh))


def _crop_tile(
    band: np.ndarray,
    tile: Tile,
    scale: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Crop a processed band to the central write region at output resolution.

    The emulator returns a smaller array (fewer pixels because GSD increases).
    We strip the overlap margins — also downscaled — to get exactly the core.
    """
    col_off = round((tile.write_window.col_off - tile.read_window.col_off) / scale)
    row_off = round((tile.write_window.row_off - tile.read_window.row_off) / scale)
    ow = _output_window(tile, scale, out_w, out_h)
    return band[row_off : row_off + ow.height, col_off : col_off + ow.width]


# ---------------------------------------------------------------------------
# Emulation config
# ---------------------------------------------------------------------------

def config_neo_to_phr() -> PipelineConfig:
    config = PipelineConfig()

    # -- Spectral -------------------------------------------------------
    config.spectral.enabled = True
    config.spectral.s2_bands = [0, 1, 2]
    config.spectral.srf_adjustment = True
    config.spectral.srf_noise_std.min = 0.005
    config.spectral.srf_noise_std.max = 0.04
    config.spectral.band_scale_factors = [1.02, 1.0, 0.98]

    # -- Spatial --------------------------------------------------------
    config.spatial.enabled = True
    config.spatial.input_gsd = _NEO_MS_GSD
    config.spatial.target_gsd.min = _PHR_MS_GSD
    config.spatial.target_gsd.max = _PHR_MS_GSD
    config.spatial.psf_sigma.min = 0.60
    config.spatial.psf_sigma.max = 0.75

    # -- Radiometric ----------------------------------------------------
    config.radiometric.enabled = True
    config.radiometric.quantization_bits = 12
    config.radiometric.sqrt_compression = False
    config.radiometric.sqrt_bands = []
    config.radiometric.saturation_threshold.min = 0.97
    config.radiometric.saturation_threshold.max = 1.0
    config.radiometric.reflectance_boost.min = 0.98
    config.radiometric.reflectance_boost.max = 1.02
    config.radiometric.reflectance_boost_prob = 0.3

    # -- Noise ----------------------------------------------------------
    config.random_noise.enabled = True
    config.random_noise.probability = 1.0
    config.random_noise.snr_db.min = 44
    config.random_noise.snr_db.max = 48
    config.random_noise.noise_type = "poisson"
    config.random_noise.poisson_weight.min = 0.4
    config.random_noise.poisson_weight.max = 0.7

    # -- Artifacts ------------------------------------------------------
    config.striping.enabled = False
    config.memory_effect.enabled = False
    config.coherent_noise.enabled = False
    config.scan_artifacts.enabled = False

    return config


# ---------------------------------------------------------------------------
# Main transform  (tiled)
# ---------------------------------------------------------------------------

def transform_neo_to_phr(
    in_path: str | Path,
    out_path: str | Path,
    tile_cfg: TileConfig = TileConfig(TILE_SIZE, TILE_OVERLAP),
) -> None:
    in_path, out_path = Path(in_path), Path(out_path)

    with rasterio.open(in_path) as src:
        neo_crs    = src.crs
        neo_bounds = src.bounds
        src_w      = src.width
        src_h      = src.height

        # Pre-compute output dimensions
        out_w = max(1, int(src_w / _GSD_RATIO))
        out_h = max(1, int(src_h / _GSD_RATIO))
        phr_transform = from_bounds(*neo_bounds, width=out_w, height=out_h)

        print(
            f"  Input  : {src_w}x{src_h}px  GSD {_NEO_MS_GSD} m\n"
            f"  Output : {out_w}x{out_h}px  GSD {_PHR_MS_GSD} m\n"
            f"  Tiles  : {math.ceil(src_w / tile_cfg.stride) * math.ceil(src_h / tile_cfg.stride)}"
            f"  ({tile_cfg.tile_size}px, overlap {tile_cfg.overlap}px)"
        )

        config   = config_neo_to_phr()
        emulator = MSSEmulator(config)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_tiles = (
            math.ceil(src_w / tile_cfg.stride)
            * math.ceil(src_h / tile_cfg.stride)
        )

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=out_h,
            width=out_w,
            count=3,
            dtype=np.uint16,
            crs=neo_crs,
            transform=phr_transform,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        ) as dst:
            dst.update_tags(
                EMULATION="PleiadesNeo_to_PHR",
                INPUT_GSD_M=str(_NEO_MS_GSD),
                OUTPUT_GSD_M=str(_PHR_MS_GSD),
                BANDS="1, 2, 3",
            )

            for tile in tqdm(
                iter_tiles(src_w, src_h, tile_cfg),
                total=n_tiles,
                desc="  tiles",
                leave=False,
            ):
                # Read tile (includes overlap padding)
                raw = src.read(
                    indexes=[1, 2, 3],
                    window=tile.read_window,
                ).astype(np.float32)

                # Normalise to [0, 1] using the sensor's known DN ceiling.
                # Using a global constant keeps radiometry consistent across
                # tile boundaries (avoids the seam artefacts of local min/max).
                norm = raw / _NEO_DN_MAX
                del raw

                # Run emulator on the tile
                phr_tile = emulator(norm)
                del norm

                # Crop overlap margins, map to output window, write
                ow = _output_window(tile, _GSD_RATIO, out_w, out_h)
                for band_idx in range(3):
                    cropped = _crop_tile(phr_tile[band_idx], tile, _GSD_RATIO, out_w, out_h)
                    dn = (cropped * 4095).clip(0, 4095).astype(np.uint16)
                    dst.write(dn, band_idx + 1, window=ow)

                del phr_tile



# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def main() -> None:
    input_folder  = Path("data/raw")
    output_folder = Path("data/raw_50_hr")
    output_folder.mkdir(parents=True, exist_ok=True)

    tif_files = list(input_folder.glob("*.tif"))
    if not tif_files:
        print(f"No .tif files found in {input_folder}")
        return

    print(f"Found {len(tif_files)} files. Starting processing...\n")

    for in_path in tqdm(tif_files, desc="Processing Images", unit="file"):
        out_path = output_folder / in_path.name
        tqdm.write(f"Processing : {in_path.name}")
        try:
            transform_neo_to_phr(in_path, out_path)
            tqdm.write(f"Saved to   : {out_path}")

            geojson_src = in_path.with_suffix(".geojson")
            if geojson_src.exists():
                geojson_dst = out_path.with_suffix(".geojson")
                shutil.copy2(geojson_src, geojson_dst)
                tqdm.write(f"Copied     : {geojson_dst.name}")
            else:
                tqdm.write(f"No GeoJSON for {in_path.name}, skipping.")

            tqdm.write("")

        except Exception as e:
            tqdm.write(f"Error processing {in_path.name}: {e}\n")
        finally:
            gc.collect()

    print("Batch processing complete.")


if __name__ == "__main__":
    main()