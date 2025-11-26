#!/usr/bin/env python3
"""
All-in-one cheese classifier: prepare data, train, and evaluate.

Usage:
    python run_all.py --coco result.json --images images/

This will:
1. Parse COCO annotations and split into train/val
2. Train EfficientNet-B0 classifier
3. Save model and show results
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from prepare_from_coco import prepare_dataset
from train import train


def main():
    parser = argparse.ArgumentParser(description='Train cheese classifier from COCO export')
    parser.add_argument('--coco', required=True, help='Path to COCO JSON (result.json)')
    parser.add_argument('--images', required=True, help='Directory containing images')
    parser.add_argument('--output', default='models', help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    
    args = parser.parse_args()
    
    data_dir = Path('data')
    
    print("=" * 60)
    print("STEP 1: Preparing dataset from COCO export")
    print("=" * 60)
    prepare_dataset(
        coco_json=args.coco,
        images_dir=args.images,
        output_dir=str(data_dir),
        val_split=0.2
    )
    
    print("\n" + "=" * 60)
    print("STEP 2: Training model")
    print("=" * 60)
    train(
        data_dir=str(data_dir),
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        patience=10,
        img_size=224
    )
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nModel saved to: {args.output}/best_model.pth")
    print(f"\nTo test on new images:")
    print(f"  python scripts/inference.py --model {args.output}/best_model.pth --image test.jpg")
    print(f"  python scripts/inference.py --model {args.output}/best_model.pth --dir test_images/")
    print(f"  python scripts/inference.py --model {args.output}/best_model.pth --video 0")


if __name__ == '__main__':
    main()
