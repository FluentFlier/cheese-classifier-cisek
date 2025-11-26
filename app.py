#!/usr/bin/env python3
"""
Gradio app for Teachable Machine Keras model.

Usage:
    1. Put keras_model.h5 and labels.txt in same folder
    2. pip install gradio tensorflow tf-keras pillow
    3. python app.py
"""

import gradio as gr
from tf_keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

# Load model and labels
print("Loading model...")
model = load_model("keras_model.h5", compile=False)

print("Loading labels...")
class_names = open("labels.txt", "r").readlines()
print(f"Classes: {[c.strip() for c in class_names]}")


def predict(image):
    """Run prediction on uploaded image."""
    if image is None:
        return {}
    
    # Create array for model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    # Convert and resize
    image = Image.fromarray(image).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Normalize to [-1, 1] (Teachable Machine format)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    
    # Predict
    prediction = model.predict(data, verbose=0)[0]
    
    # Return as dict for Gradio
    results = {}
    for i, class_name in enumerate(class_names):
        label = class_name.strip().split(" ", 1)[-1] if " " in class_name else class_name.strip()
        results[label] = float(prediction[i])
    
    return results


# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Label(num_top_classes=len(class_names), label="Prediction"),
    title="Cheese Classifier",
    description="Upload an image to classify",
)

if __name__ == "__main__":
    demo.launch()