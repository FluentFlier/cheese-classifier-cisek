#!/usr/bin/env python3
"""
FastAPI server for cheese classification.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import io
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image


# Response models
class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    probabilities: dict
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    classes: List[str]


# Global model
MODEL = None
CLASS_NAMES = []
TRANSFORM = None
DEVICE = None


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(model_path: str = "models/best_model.pth"):
    """Load model on startup."""
    global MODEL, CLASS_NAMES, TRANSFORM, DEVICE
    
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")
    
    if not Path(model_path).exists():
        print(f"Warning: Model not found at {model_path}")
        return False
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    CLASS_NAMES = checkpoint.get('class_names', [])
    config = checkpoint.get('config', {})
    img_size = config.get('model', {}).get('img_size', 224)
    
    MODEL = models.efficientnet_b0(weights=None)
    in_features = MODEL.classifier[1].in_features
    MODEL.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features, len(CLASS_NAMES))
    )
    
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    TRANSFORM = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"Model loaded. Classes: {CLASS_NAMES}")
    return True


def predict_image(image: Image.Image) -> dict:
    """Run prediction on single image."""
    start = time.time()
    
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    
    inference_time = (time.time() - start) * 1000
    
    return {
        'class_name': CLASS_NAMES[predicted.item()],
        'confidence': confidence.item(),
        'probabilities': {name: probs[0, i].item() for i, name in enumerate(CLASS_NAMES)},
        'inference_time_ms': inference_time
    }


# FastAPI app
app = FastAPI(
    title="Cheese Classification API",
    description="API for classifying cheese images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if MODEL is not None else "no model",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else "none",
        "classes": CLASS_NAMES
    }


@app.get("/classes")
async def get_classes():
    """Get available classes."""
    return {"classes": CLASS_NAMES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Classify single image."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        result = predict_image(image)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Classify multiple images."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    predictions = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            result = predict_image(image)
            result['filename'] = file.filename
            predictions.append(result)
        except Exception as e:
            predictions.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        'predictions': predictions,
        'total_time_ms': (time.time() - start) * 1000
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
