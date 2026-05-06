#!/usr/bin/env python3
"""
scripts/background.py
---------------------
Curates the tiled dataset by moving excess background (empty) tiles into a 'moved' folder.
Ensures that background images make up a strict maximum percentage of the total dataset.
Only processes the 'train' and 'val' splits.
"""

import argparse
import random
import shutil
from pathlib import Path

def is_background(label_path: Path) -> bool:
    """
    Determines if an image is a background based on its label file.
    In YOLO, missing files or completely empty files are backgrounds.
    """
    if not label_path.exists():
        return True
    with open(label_path, 'r') as f:
        content = f.read().strip()
    return len(content) == 0

def process_split(split: str, dataset_dir: Path, target_bg_ratio: float, seed: int = 42):
    print(f"\n[{split.upper()} SPLIT]")
    
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split
    
    if not img_dir.exists():
        print(f"  Directory not found: {img_dir} - Skipping.")
        return

    # Create 'moved' backup directories
    moved_img_dir = dataset_dir / "moved" / "images" / split
    moved_lbl_dir = dataset_dir / "moved" / "labels" / split
    moved_img_dir.mkdir(parents=True, exist_ok=True)
    moved_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images (Assuming .tif based on your previous logs, but grabs common formats)
    image_paths = []
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(img_dir.glob(ext))

    positives = []
    backgrounds = []

    for img_path in image_paths:
        label_path = lbl_dir / f"{img_path.stem}.txt"
        if is_background(label_path):
            backgrounds.append((img_path, label_path))
        else:
            positives.append((img_path, label_path))

    num_pos = len(positives)
    num_bg = len(backgrounds)
    total_original = num_pos + num_bg

    if num_pos == 0:
        print("  ⚠️ No positive images found! Skipping to prevent deleting everything.")
        return

    # Calculate how many backgrounds to KEEP based on the desired ratio
    # Math: ratio = bg_keep / (positives + bg_keep)
    # bg_keep = positives * (ratio / (1 - ratio))
    bg_keep = int(num_pos * (target_bg_ratio / (1.0 - target_bg_ratio)))
    
    print(f"  Original Totals: {total_original} images ({num_pos} with boats, {num_bg} empty)")
    print(f"  Target Background Ratio: {target_bg_ratio * 100:.1f}%")

    if num_bg <= bg_keep:
        print(f"  ✅ Dataset is already at or below target ratio (Keep: {bg_keep}, Actual: {num_bg}). Nothing to move.")
        return

    num_to_move = num_bg - bg_keep
    print(f"  Calculated limits: Keep {bg_keep} empty tiles, Move {num_to_move} empty tiles.")

    # Shuffle backgrounds to ensure random spatial distribution of kept backgrounds
    random.seed(seed)
    random.shuffle(backgrounds)

    backgrounds_to_move = backgrounds[bg_keep:]

    moved_count = 0
    for img_path, lbl_path in backgrounds_to_move:
        # Move image
        shutil.move(str(img_path), str(moved_img_dir / img_path.name))
        
        # Move label if it exists (it might be a 0-byte file)
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(moved_lbl_dir / lbl_path.name))
            
        moved_count += 1

    print(f"  🚀 Successfully moved {moved_count} background tiles to: {moved_img_dir}")
    print(f"  New Dataset Size for {split}: {num_pos + bg_keep} ({num_pos} boats, {bg_keep} empty)")


def main():
    parser = argparse.ArgumentParser(description="Filter out excess background tiles.")
    parser.add_argument(
        "--data", 
        type=str, 
        default="/home/thomas/Documents/code/pleiades-boat-detection/data/number/tiled/train/8", 
        help="Path to the tiled dataset root directory."
    )
    parser.add_argument(
        "--ratio", 
        type=float, 
        default=0.15, 
        help="Target ratio of background images in the final dataset (0.15 = 15%)."
    )
    args = parser.parse_args()

    dataset_path = Path(args.data)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist.")
        return

    # ONLY process train and val. Skip test.
    splits_to_process = ['']
    
    for split in splits_to_process:
        process_split(split, dataset_path, args.ratio)

    print("\n🎉 Background curation complete! Don't forget to delete your .cache files before running YOLO again.")


if __name__ == "__main__":
    main()