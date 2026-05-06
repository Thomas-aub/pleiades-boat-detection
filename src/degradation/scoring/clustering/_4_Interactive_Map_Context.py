import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import PickEvent
import rasterio
from rasterio.plot import reshape_as_image
from pathlib import Path
import os

# --- CONFIGURATION PATHS ---
RAW_DATA_BASE = Path("data/scoring/diff")
TILED_DATA_BASE = Path("data/scoring/tiled")

def normalize_for_display(img_data: np.ndarray) -> np.ndarray:
    """Normalizes raw image data to a displayable uint8 RGB format."""
    if img_data.ndim == 3 and img_data.shape[2] > 3:
        img_data = img_data[:, :, :3]  
    elif img_data.ndim == 2:
        img_data = np.stack([img_data] * 3, axis=-1)  

    img_data = img_data.astype(np.float32)
    p2, p98 = np.percentile(img_data, (2, 98))
    
    if p98 > p2:
        img_norm = np.clip((img_data - p2) / (p98 - p2), 0, 1)
        return (img_norm * 255).astype(np.uint8)
    return img_data.astype(np.uint8)


class UltimateInteractivePlot:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.csv_file = TILED_DATA_BASE / folder_name / "clustering_results.csv"
        
        if not self.csv_file.exists():
            raise FileNotFoundError(f"Missing {self.csv_file}. Please run Step 3 first.")
            
        self.df = pd.read_csv(self.csv_file)
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        self.fig_chip = None
        self.fig_ctx = None
        
        self.setup_main_plot()

    def setup_main_plot(self) -> None:
        self.scatter = self.ax.scatter(
            self.df['tsne_1'], self.df['tsne_2'], 
            c=self.df['cluster'], cmap='tab20', 
            s=15, alpha=0.7, picker=5
        )
        
        self.ax.legend(*self.scatter.legend_elements(), title="Clusters")
        self.ax.set_title(f'Interactive t-SNE ({self.folder_name}): Click a point to see details', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event: PickEvent) -> None:
        ind = event.ind[0]
        row = self.df.iloc[ind]
        chip_path = Path(row['chip_path'])

        if not chip_path.exists():
            print(f"Tile image not found: {chip_path}")
            return

        # 1. Read the tile and extract its embedded provenance metadata tags
        with rasterio.open(chip_path) as tile_src:
            img_raw = tile_src.read()
            tags = tile_src.tags()
            
            source_tif_name = tags.get('source_tif')
            x_off = int(float(tags.get('col_off', 0)))
            y_off = int(float(tags.get('row_off', 0)))
            tile_size = int(float(tags.get('tile_size', img_raw.shape[1])))

        # 2. Show the Tile
        if self.fig_chip and plt.fignum_exists(self.fig_chip.number):
            plt.close(self.fig_chip)

        img_display = reshape_as_image(img_raw)
        img_display = normalize_for_display(img_display)

        self.fig_chip, ax_chip = plt.subplots(figsize=(5, 5))
        ax_chip.imshow(img_display)
        ax_chip.set_title(f"Tile: {chip_path.name}\nCluster: {row['cluster']}")
        ax_chip.axis('off')
        self.fig_chip.canvas.manager.window.move(100, 100)
        plt.show(block=False)

        # 3. Show the Global Context
        if self.fig_ctx and plt.fignum_exists(self.fig_ctx.number):
            plt.close(self.fig_ctx)

        if not source_tif_name:
            print("No source_tif tag found in tile. Cannot show context.")
            return

        raw_tif_path = RAW_DATA_BASE / self.folder_name / source_tif_name

        if not raw_tif_path.exists():
            print(f"Context original image not found: {raw_tif_path}")
            return

        with rasterio.open(raw_tif_path) as src:
            window_size = 2500 # Larger window to accommodate 640x640 or 1024x1024 tiles
            
            # Center the context window around the center of our tile
            center_x = x_off + (tile_size // 2)
            center_y = y_off + (tile_size // 2)
            
            win_x = center_x - (window_size // 2)
            win_y = center_y - (window_size // 2)
            
            window = rasterio.windows.Window(win_x, win_y, window_size, window_size)
            
            # Read context data using direct pixel coordinate translation
            img_context = src.read(window=window, boundless=True, fill_value=0)
            img_show = reshape_as_image(img_context)
            img_show = normalize_for_display(img_show)

            self.fig_ctx, ax_ctx = plt.subplots(figsize=(8, 8))
            ax_ctx.imshow(img_show)
            
            # Draw a bounding box indicating exactly where the tile is inside the context
            rect_x = (window_size // 2) - (tile_size // 2)
            rect_y = (window_size // 2) - (tile_size // 2)
            rect = patches.Rectangle(
                (rect_x, rect_y), tile_size, tile_size, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax_ctx.add_patch(rect)
            ax_ctx.set_title(f"Global Context: {source_tif_name}\nOriginal Location: X={x_off}, Y={y_off}")
            
            self.fig_ctx.canvas.manager.window.move(100, 900)
            plt.show(block=False)

if __name__ == "__main__":
    # --- CHOOSE YOUR FOLDER HERE ---
    # Options: "Gen", "PL", or "PLN"
    FOLDER_TO_EXPLORE = "PLN"  
    
    print(f"Launching interactive plot for folder: {FOLDER_TO_EXPLORE}...")
    print(f"(Change FOLDER_TO_EXPLORE variable at the bottom of the script to view other datasets)")
    
    try:
        app = UltimateInteractivePlot(FOLDER_TO_EXPLORE)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")