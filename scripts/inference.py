#!/usr/bin/env python3
"""
Run inference on images.

Usage:
    python scripts/inference.py --model models/best_model.pth --image test.jpg
    python scripts/inference.py --model models/best_model.pth --dir test_images/
    python scripts/inference.py --model models/best_model.pth --video 0
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(model_path: str, device: torch.device):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    class_names = checkpoint.get('class_names', [])
    config = checkpoint.get('config', {})
    img_size = config.get('model', {}).get('img_size', 224)
    
    # Recreate model
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features, len(class_names))
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names, img_size


def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_image(model, image: Image.Image, transform, class_names, device) -> dict:
    """Predict single image."""
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    
    return {
        'class': class_names[predicted.item()],
        'confidence': confidence.item(),
        'probabilities': {name: probs[0, i].item() for i, name in enumerate(class_names)}
    }


def predict_single(model_path: str, image_path: str):
    """Predict class for single image."""
    device = get_device()
    model, class_names, img_size = load_model(model_path, device)
    transform = get_transform(img_size)
    
    image = Image.open(image_path).convert('RGB')
    result = predict_image(model, image, transform, class_names, device)
    
    print(f"Image: {image_path}")
    print(f"Prediction: {result['class']} ({result['confidence']:.1%})")
    print(f"Probabilities: {result['probabilities']}")
    
    return result


def predict_directory(model_path: str, image_dir: str, output_file: str = None):
    """Predict classes for all images in directory."""
    device = get_device()
    model, class_names, img_size = load_model(model_path, device)
    transform = get_transform(img_size)
    
    image_dir = Path(image_dir)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]
    
    results = []
    for img_path in image_files:
        image = Image.open(img_path).convert('RGB')
        result = predict_image(model, image, transform, class_names, device)
        result['file'] = str(img_path.name)
        results.append(result)
        print(f"{img_path.name}: {result['class']} ({result['confidence']:.1%})")
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Summary
    counts = {}
    for r in results:
        counts[r['class']] = counts.get(r['class'], 0) + 1
    print(f"\nSummary: {counts}")
    
    return results


def predict_video(model_path: str, video_source=0, display: bool = True):
    """Real-time prediction on video."""
    import cv2
    
    device = get_device()
    model, class_names, img_size = load_model(model_path, device)
    transform = get_transform(img_size)
    
    cap = cv2.VideoCapture(int(video_source) if str(video_source).isdigit() else video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    print("Running inference... Press 'q' to quit")
    fps_list = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start = time.time()
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            result = predict_image(model, image, transform, class_names, device)
            
            fps = 1.0 / (time.time() - start)
            fps_list.append(fps)
            
            if display:
                label = f"{result['class']}: {result['confidence']:.1%}"
                color = (0, 255, 0) if result['confidence'] > 0.8 else (0, 165, 255)
                
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Cheese Classifier', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        if fps_list:
            print(f"\nAverage FPS: {np.mean(fps_list):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--dir', help='Directory of images')
    parser.add_argument('--video', nargs='?', const=0, help='Video source')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    if args.image:
        predict_single(args.model, args.image)
    elif args.dir:
        predict_directory(args.model, args.dir, args.output)
    elif args.video is not None:
        predict_video(args.model, args.video, not args.no_display)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
