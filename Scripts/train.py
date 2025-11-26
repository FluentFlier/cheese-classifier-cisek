#!/usr/bin/env python3
"""
Cheese Classification Training Script

Uses EfficientNet-B0 (BSD license) for binary classification:
- white_cubed
- nothing

Features:
- Transfer learning from ImageNet
- Data augmentation for robustness
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


# Class names (must match folder names in data/)
# Alphabetically sorted to match ImageFolder behavior
CLASS_NAMES = ['cheese', 'no_cheese']


def get_transforms(is_training: bool = True, img_size: int = 224):
    """Get image transforms for training or inference."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 model with custom classifier."""
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Per-class accuracy
    class_correct = {c: 0 for c in range(len(CLASS_NAMES))}
    class_total = {c: 0 for c in range(len(CLASS_NAMES))}
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    class_acc = {}
    for c in range(len(CLASS_NAMES)):
        if class_total[c] > 0:
            class_acc[CLASS_NAMES[c]] = class_correct[c] / class_total[c]
        else:
            class_acc[CLASS_NAMES[c]] = 0.0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'class_accuracy': class_acc
    }


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 10,
    img_size: int = 224
):
    """
    Main training function.
    
    Args:
        data_dir: Directory with train/ and val/ subdirs
        output_dir: Directory to save model and logs
        epochs: Maximum training epochs
        batch_size: Batch size
        lr: Initial learning rate
        patience: Early stopping patience
        img_size: Input image size
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_dataset = datasets.ImageFolder(
        data_dir / 'train',
        transform=get_transforms(is_training=True, img_size=img_size)
    )
    val_dataset = datasets.ImageFolder(
        data_dir / 'val',
        transform=get_transforms(is_training=False, img_size=img_size)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Verify class order matches expected
    if train_dataset.classes != CLASS_NAMES:
        print(f"Warning: Class order {train_dataset.classes} differs from expected {CLASS_NAMES}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model
    model = create_model(num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    no_improve_count = 0
    history = {'train': [], 'val': []}
    
    print(f"\nStarting training for up to {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Record history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Per-class: {val_metrics['class_accuracy']}")
        print(f"  Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            no_improve_count = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'class_names': CLASS_NAMES,
            }, output_dir / 'best_model.pth')
            print(f"  ** Saved new best model (acc: {best_val_acc:.4f})")
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': CLASS_NAMES,
    }, output_dir / 'final_model.pth')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = {
        'class_names': CLASS_NAMES,
        'img_size': img_size,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_acc,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train cheese classifier')
    parser.add_argument('--data', default='data', help='Data directory with train/val splits')
    parser.add_argument('--output', default='models', help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=50, help='Max training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        img_size=args.img_size
    )


if __name__ == '__main__':
    main()
