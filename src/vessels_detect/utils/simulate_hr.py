# pip install git+https://github.com/JulioContrerasH/satharmony

"""Pléiades Neo → Pléiades (PHR 1A/1B) sensor emulation.

Transforms Pléiades Neo PMS imagery to the radiometric and spatial
characteristics of the Pléiades (PHR 1A/1B) PMS product.

--------------------------------------------------------------------
Pléiades (PHR 1A/1B) — Airbus Pléiades User Guide (PHR-UG):
* GSD: 0.5 m (Pan-sharpened PMS), 2.0 m (raw MS)
* Spectral bands — PHR-UG §2.2.2 & Table 1.2:
    B0 Blue  : 430–550 nm
    B1 Green : 500–620 nm
    B2 Red   : 590–710 nm
    B3 NIR   : 740–940 nm
* Quantization: 12-bit, DN range 0–4095 — PHR-UG §2.4.1
* MTF @ Nyquist (in-flight, after ground MTF correction) — PHR-UG §2.3:
    PAN : 0.15  →  σ ≈ 0.620 px  (Gaussian PSF model, see config below)
    MS  : 0.30  →  σ ≈ 0.494 px  (larger pixels → better MTF@Nyquist)
* SNR (in-flight) — PHR-UG §2.3:
    RGB channels : ~150   → 20·log10(150) ≈ 43.5 dB
    NIR channel  : ~190   → 20·log10(190) ≈ 45.6 dB

Pléiades Neo (3/4) — Airbus Pléiades Neo User Guide v3 Oct 2021 (NEO-UG):
* GSD: 0.3 m (PAN/PMS), 1.2 m (4-band MS) — NEO-UG Table 4
* Spectral bands — NEO-UG §2.1 Fig. 9:
    B0 Red  : 620–690 nm  (band order in 4-band file: R, G, B, NIR — NEO-UG Table 28)
    B1 Green: 530–590 nm
    B2 Blue : 450–520 nm
    B3 NIR  : 770–880 nm
* Quantization: 12-bit — NEO-UG Table 4
* MTF PAN @ Nyquist: > 0.15 — NEO-UG Table 5
* SNR PAN @ 100 W·m⁻²·sr⁻¹·μm⁻¹: > 100 — NEO-UG Table 5
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
# GSD constants  (NEO-UG Table 4; PHR-UG §2.2)
# ---------------------------------------------------------------------------

_PHR_MS_GSD = 0.5   # PHR PMS product resolution — PHR-UG §2.2
_NEO_MS_GSD = 0.3   # Neo PMS product resolution  — NEO-UG Table 4
_GSD_RATIO  = _PHR_MS_GSD / _NEO_MS_GSD   # 5/3 ≈ 1.667 — output has fewer pixels

# 12-bit ADC on both sensors; normalise to the physical DN ceiling.
# Global constant avoids per-tile min/max rescaling artefacts at tile seams.
_NEO_DN_MAX = 4095.0   # NEO-UG Table 4 / PHR-UG §2.4.1

# ---------------------------------------------------------------------------
# Tiling constants  (tune to available RAM)
# ---------------------------------------------------------------------------

TILE_SIZE    = 2048   # input pixels per tile side
TILE_OVERLAP = 0      # overlap pixels (absorbs PSF/blur edge artefacts)


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
    """Build the Neo → PHR emulation configuration.

    Every physical parameter is justified against the official Airbus user
    guides (abbreviated below):
      PHR-UG : Pléiades User Guide (PHR 1A/1B)
      NEO-UG : Pléiades Neo User Guide, Early Version 3, Oct 2021

    PSF sigma derivation (Gaussian MTF model):
        MTF(f) = exp(−π²σ²f²/2),   f in cycles/pixel,  Nyquist = 0.5 cyc/px
        ⟹  σ = sqrt(−2·ln(MTF_Nyq) / π²)
    """
    config = PipelineConfig()

    # -- Spectral -------------------------------------------------------

    # Neo 4-band MS file band order: R(0), G(1), B(2), NIR(3) — NEO-UG Table 28.
    # Selecting [0, 1, 2] retains R, G, B, matching the PHR 3-band PMS-N product.
    config.spectral.enabled = True
    config.spectral.s2_bands = [0, 1, 2]

    # PHR bands are wider than Neo's — SRF adjustment is required:
    #   Neo Blue : 450–520 nm  vs  PHR Blue : 430–550 nm  (NEO-UG §2.1; PHR-UG §2.2.2)
    #   Neo Green: 530–590 nm  vs  PHR Green: 500–620 nm
    #   Neo Red  : 620–690 nm  vs  PHR Red  : 590–710 nm
    config.spectral.srf_adjustment = True

    # Residual SRF uncertainty after band-matching correction.
    # Absolute radiometric calibration uncertainty ≤ 5% (NEO-UG §A.2.2.1,
    # Band_Radiance MEASURE_UNCERTAINTY = 5). Residual after SRF correction
    # is conservatively bounded at ≤ 2% per band.
    config.spectral.srf_noise_std.min = 0.005
    config.spectral.srf_noise_std.max = 0.02

    # No systematic per-band scale offset: public documentation reports no
    # known gain imbalance between Neo RGB and PHR RGB output channels.
    # A deterministic ramp would introduce a colour cast with no physical
    # grounding and was identified as a likely contributor to the KID gap.
    config.spectral.band_scale_factors = [1.0, 1.0, 1.0]

    # -- Spatial --------------------------------------------------------

    # Neo PMS GSD: 0.3 m — NEO-UG Table 4.
    config.spatial.enabled = True
    config.spatial.input_gsd = _NEO_MS_GSD

    # PHR PMS GSD: 0.5 m — PHR-UG §2.2.
    config.spatial.target_gsd.min = _PHR_MS_GSD
    config.spatial.target_gsd.max = _PHR_MS_GSD

    # PSF sigma derived from PHR PAN MTF@Nyquist (PHR-UG §2.3, in-flight
    # measurement after ground MTF correction):
    #   MTF_Nyq(PAN) = 0.15  →  σ = sqrt(−2·ln(0.15)/π²) ≈ 0.620 px
    # The PMS product at 0.5 m is spatially dominated by the PAN channel,
    # so the PAN MTF drives the PSF. A small range [0.60, 0.64] captures
    # inter-image variability (off-nadir, attitude residuals) while staying
    # within the documented MTF specification.
    config.spatial.psf_sigma.min = 0.60
    config.spatial.psf_sigma.max = 0.64

    # -- Radiometric ----------------------------------------------------

    # Both Neo and PHR use 12-bit ADC, DN ∈ [0, 4095] — NEO-UG Table 4,
    # PHR-UG §2.4.1. Linear encoding; no sqrt compression on either sensor.
    config.radiometric.enabled = True
    config.radiometric.quantization_bits = 12
    config.radiometric.sqrt_compression = False
    config.radiometric.sqrt_bands = []

    # Pixel saturation: PHR AGC keeps acquisitions within 97–100% of full
    # well capacity under nominal conditions — PHR-UG §2.4.1.
    config.radiometric.saturation_threshold.min = 0.97
    config.radiometric.saturation_threshold.max = 1.0

    # Scene-to-scene reflectance variability within the ≤5% absolute
    # calibration envelope (NEO-UG §A.2.2.1). ±2% boost covers realistic
    # inter-image calibration spread without exceeding the certified bound.
    config.radiometric.reflectance_boost.min = 0.98
    config.radiometric.reflectance_boost.max = 1.02
    config.radiometric.reflectance_boost_prob = 0.3

    # -- Noise ----------------------------------------------------------

    # Poisson (shot noise) is the dominant noise source in pushbroom VHR
    # sensors at typical scene radiance — physically correct for PHR.
    config.random_noise.enabled = True
    config.random_noise.probability = 1.0
    config.random_noise.noise_type = "poisson"

    # PHR in-flight MS SNR — PHR-UG §2.3:
    #   RGB channels: ~150  →  20·log10(150) ≈ 43.5 dB
    #   NIR channel : ~190  →  20·log10(190) ≈ 45.6 dB
    # We emulate RGB only (s2_bands=[0,1,2]), so the target is ~43.5 dB.
    # The range [42, 44] accounts for scene-dependent SNR degradation
    # (low-radiance / off-nadir acquisitions) while capping well below
    # Neo's ~48 dB — ensuring noise is always added in the emulation.
    config.random_noise.snr_db.min = 42
    config.random_noise.snr_db.max = 44

    # Poisson weight: fraction of total noise variance attributable to
    # photon shot noise. For a well-exposed 12-bit pushbroom sensor,
    # shot noise accounts for ~50–70% of total variance, with the
    # remainder from read noise and quantisation — Kim et al. 2025;
    # Wang et al. 2021.
    config.random_noise.poisson_weight.min = 0.5
    config.random_noise.poisson_weight.max = 0.7

    # -- Artifacts (not present in standard PHR ortho products) ---------
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
                # tile boundaries (avoids seam artefacts of local min/max).
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
    input_folder  = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/raw")
    output_folder = Path("/home/thomas/Documents/code/pleiades-boat-detection/data/raw50")
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