#!/usr/bin/env python3
"""
Gradio demo app for cheese classifier.

Usage:
    python app.py
    python app.py --model models/best_model.pth
"""

import argparse
from pathlib import Path

import gradio as gr
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


def load_model(model_path: str):
    """Load model for inference."""
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    
    class_names = checkpoint.get('class_names', [])
    config = checkpoint.get('config', {})
    img_size = config.get('model', {}).get('img_size', 224)
    
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features, len(class_names))
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, class_names, transform, device


def create_app(model_path: str):
    """Create Gradio app."""
    print(f"Loading model from {model_path}...")
    model, class_names, transform, device = load_model(model_path)
    print(f"Classes: {class_names}")
    
    def predict(image):
        if image is None:
            return {}
        
        image = Image.fromarray(image).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
        
        return {class_names[i]: float(probs[0, i]) for i in range(len(class_names))}
    
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Upload Image"),
        outputs=gr.Label(num_top_classes=len(class_names), label="Prediction"),
        title="Cheese Classifier",
        description="Upload an image to classify cheese type",
        examples=None,
    )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description='Run Gradio demo')
    parser.add_argument('--model', default='models/best_model.pth', help='Model path')
    parser.add_argument('--port', type=int, default=7860, help='Port')
    parser.add_argument('--share', action='store_true', help='Create public link')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Train a model first: python scripts/train.py --data data/ --output models/")
        return
    
    demo = create_app(args.model)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
