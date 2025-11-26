#!/usr/bin/env python3
"""
Split image folders into train/val sets.

Input structure:
    images/
    ├── cheese/
    └── no_cheese/

Output structure:
    data/
    ├── train/
    │   ├── cheese/
    │   └── no_cheese/
    └── val/
        ├── cheese/
        └── no_cheese/

Usage:
    python split_folders.py --input images/ --output data/
"""

import argparse
import shutil
import random
from pathlib import Path


def split_folders(input_dir: str, output_dir: str, val_split: float = 0.2, seed: int = 42):
    random.seed(seed)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find class folders
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"Error: No class folders found in {input_dir}")
        print("Expected structure:")
        print("  images/")
        print("    cheese/")
        print("    no_cheese/")
        return
    
    print(f"Found classes: {[d.name for d in class_dirs]}")
    
    # Create output structure
    for split in ['train', 'val']:
        for class_dir in class_dirs:
            (output_dir / split / class_dir.name).mkdir(parents=True, exist_ok=True)
    
    # Split each class
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    stats = {'train': {}, 'val': {}}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]
        
        print(f"  {class_name}: {len(images)} images")
        
        random.shuffle(images)
        n_val = max(1, int(len(images) * val_split))
        
        val_images = images[:n_val]
        train_images = images[n_val:]
        
        stats['train'][class_name] = len(train_images)
        stats['val'][class_name] = len(val_images)
        
        for img in train_images:
            shutil.copy2(img, output_dir / 'train' / class_name / img.name)
        
        for img in val_images:
            shutil.copy2(img, output_dir / 'val' / class_name / img.name)
    
    print(f"\nDataset created in {output_dir}:")
    print(f"  train/")
    for c, n in stats['train'].items():
        print(f"    {c}: {n}")
    print(f"  val/")
    for c, n in stats['val'].items():
        print(f"    {c}: {n}")
    
    print(f"\nNext: python scripts/train.py --data {output_dir} --output models/")


def main():
    parser = argparse.ArgumentParser(description='Split image folders into train/val')
    parser.add_argument('--input', required=True, help='Input directory with class folders')
    parser.add_argument('--output', default='data', help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    split_folders(args.input, args.output, args.val_split, args.seed)


if __name__ == '__main__':
    main()
