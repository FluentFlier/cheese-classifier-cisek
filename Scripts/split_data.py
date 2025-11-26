#!/usr/bin/env python3
"""
Split image folders into train/val sets.

Usage:
    python scripts/split_data.py --input images/ --output data/
"""

import argparse
import shutil
import random
import yaml
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {"training": {"val_split": 0.2}}


def split_folders(input_dir: str, output_dir: str, val_split: float = None, seed: int = 42):
    random.seed(seed)
    
    config = load_config()
    if val_split is None:
        val_split = config.get("training", {}).get("val_split", 0.2)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find class folders
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not class_dirs:
        print(f"Error: No class folders found in {input_dir}")
        print("Expected structure:")
        print("  images/")
        print("    white_cubed/")
        print("    nothing/")
        print("    ...")
        return False
    
    print(f"Found {len(class_dirs)} classes: {sorted([d.name for d in class_dirs])}")
    
    # Create output structure
    for split in ['train', 'val']:
        for class_dir in class_dirs:
            (output_dir / split / class_dir.name).mkdir(parents=True, exist_ok=True)
    
    # Split each class
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    stats = {'train': {}, 'val': {}}
    total_train, total_val = 0, 0
    
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]
        
        if len(images) == 0:
            print(f"  Warning: {class_name}/ has no images, skipping")
            continue
        
        random.shuffle(images)
        n_val = max(1, int(len(images) * val_split))
        
        val_images = images[:n_val]
        train_images = images[n_val:]
        
        stats['train'][class_name] = len(train_images)
        stats['val'][class_name] = len(val_images)
        total_train += len(train_images)
        total_val += len(val_images)
        
        for img in train_images:
            shutil.copy2(img, output_dir / 'train' / class_name / img.name)
        
        for img in val_images:
            shutil.copy2(img, output_dir / 'val' / class_name / img.name)
        
        print(f"  {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    print(f"\nDataset created: {total_train} train, {total_val} val")
    print(f"Output: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Split image folders into train/val')
    parser.add_argument('--input', default='images', help='Input directory with class folders')
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--val-split', type=float, default=None, help='Validation split (default from config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Clean output directory
    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"Cleaning existing {output_dir}...")
        shutil.rmtree(output_dir)
    
    success = split_folders(args.input, args.output, args.val_split, args.seed)
    if success:
        print(f"\nNext: python scripts/train.py --data {args.output} --output models/")


if __name__ == '__main__':
    main()
