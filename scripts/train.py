#!/usr/bin/env python3
"""
Train cheese classifier.

Usage:
    python scripts/train.py --data data/ --output models/
"""

import argparse
import json
import time
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_transforms(config, is_training: bool = True):
    """Get image transforms."""
    img_size = config.get("model", {}).get("img_size", 224)
    aug = config.get("augmentation", {})
    
    if is_training:
        transform_list = [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
        ]
        
        if aug.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if aug.get("vertical_flip", False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.3))
        if aug.get("rotation", 0) > 0:
            transform_list.append(transforms.RandomRotation(aug["rotation"]))
        
        cj = aug.get("color_jitter", {})
        if cj:
            transform_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.05)
            ))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    
    return transforms.Compose(transform_list)


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create EfficientNet-B0 model."""
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
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
    
    return {'loss': running_loss / total, 'accuracy': correct / total}


def validate(model, loader, criterion, device, class_names):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = {c: 0 for c in range(len(class_names))}
    class_total = {c: 0 for c in range(len(class_names))}
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for pred, label in zip(predicted, labels):
                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1
    
    class_acc = {}
    for c in range(len(class_names)):
        if class_total[c] > 0:
            class_acc[class_names[c]] = class_correct[c] / class_total[c]
    
    return {
        'loss': running_loss / total,
        'accuracy': correct / total,
        'class_accuracy': class_acc
    }


def train(data_dir: str, output_dir: str, config: dict = None):
    """Main training function."""
    if config is None:
        config = load_config()
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training params
    train_cfg = config.get("training", {})
    epochs = train_cfg.get("epochs", 50)
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("lr", 0.001)
    patience = train_cfg.get("patience", 10)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Data
    train_dataset = datasets.ImageFolder(
        data_dir / 'train',
        transform=get_transforms(config, is_training=True)
    )
    val_dataset = datasets.ImageFolder(
        data_dir / 'val',
        transform=get_transforms(config, is_training=False)
    )
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes ({num_classes}): {class_names}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4 if device.type != 'mps' else 0,
        pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4 if device.type != 'mps' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    # Model
    model = create_model(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {'train': [], 'val': []}
    
    print(f"\nTraining for up to {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        start = time.time()
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device, class_names)
        
        scheduler.step(val_metrics['loss'])
        
        history['train'].append(train_metrics)
        history['val'].append({k: v for k, v in val_metrics.items() if k != 'class_accuracy'})
        
        elapsed = time.time() - start
        
        print(f"Epoch {epoch + 1}/{epochs} ({elapsed:.1f}s)")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'class_names': class_names,
                'config': config,
            }, output_dir / 'best_model.pth')
            print(f"  ** Saved best model (acc: {best_val_acc:.4f})")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Save final model and history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'config': config,
    }, output_dir / 'final_model.pth')
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'class_names': class_names,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_acc,
            'timestamp': datetime.now().isoformat(),
            **config
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Best accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {output_dir}/best_model.pth")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train cheese classifier')
    parser.add_argument('--data', default='data', help='Data directory')
    parser.add_argument('--output', default='models', help='Output directory')
    parser.add_argument('--epochs', type=int, help='Override epochs from config')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Override config with command line args
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["lr"] = args.lr
    
    train(args.data, args.output, config)


if __name__ == '__main__':
    main()
