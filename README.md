# Cheese Inspection System - Production

Computer vision system for CISEK cheese quality inspection.

## Project Structure

```
cheese-inspection/
├── images/                     # RAW IMAGES - organize by class
│   ├── white_cubed/
│   ├── yellow_cubed/
│   ├── shredded/
│   ├── nothing/
│   └── cleaning/
├── data/                       # GENERATED - train/val splits
│   ├── train/
│   └── val/
├── models/                     # GENERATED - saved models
├── results/                    # GENERATED - evaluation reports
│   ├── report.html             # Interactive HTML report with images
│   ├── predictions.csv
│   ├── errors.csv
│   ├── label_studio_import.json
│   └── summary.json
├── scripts/
│   ├── split_data.py          # Split images into train/val
│   ├── train.py               # Training script
│   ├── inference.py           # Run predictions
│   └── evaluate.py            # Evaluate model & export reports
├── api/
│   ├── main.py                # FastAPI server
│   └── requirements.txt       # API dependencies
├── docker/
│   ├── Dockerfile.train       # Training container
│   ├── Dockerfile.api         # Deployment container
│   └── docker-compose.yml     # Orchestration
├── configs/
│   └── config.yaml            # Training config
├── tests/
│   └── test_inference.py      # Basic tests
├── app.py                     # Gradio demo app
├── requirements.txt           # Main dependencies
└── README.md
```

---

## Quick Start

### 1. Setup

```bash
# Clone and enter project
cd cheese-inspection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Organize Images

Put your images in the `images/` folder by class:
```
images/
├── white_cubed/   # ~50-200 images
├── nothing/       # ~50-200 images
└── ...
```

### 3. Train

```bash
# Split into train/val
python scripts/split_data.py --input images/ --output data/

# Train model
python scripts/train.py --data data/ --output models/
```

### 4. Test

```bash
# Single image
python scripts/inference.py --model models/best_model.pth --image test.jpg

# Gradio demo
python app.py

# FastAPI server
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs
```

### 5. Evaluate & Review

Evaluate model performance and export results for review:

```bash
# Run evaluation on your images
python scripts/evaluate.py --model models/best_model.pth --images images/ --output results/
```

**Outputs:**
- `results/report.html` - **Interactive HTML report with image thumbnails** (open in browser)
- `results/predictions.csv` - All predictions with confidence scores
- `results/errors.csv` - Only incorrect predictions for quick review
- `results/label_studio_import.json` - Import into Label Studio for corrections
- `results/summary.json` - Overall and per-class accuracy statistics

**View HTML Report:**
```bash
open results/report.html  # Mac
xdg-open results/report.html  # Linux
start results/report.html  # Windows
```

The HTML report includes:
- Visual summary with accuracy stats
- Per-class accuracy bars
- Image thumbnails with predictions
- Tabs for "Errors Only" and "All Results"
- Click images to view full size
- Color-coded correct/incorrect predictions

**Label Studio Workflow:**

1. **Run evaluation** to get predictions:
   ```bash
   python scripts/evaluate.py
   ```

2. **Import into Label Studio:**
   - Create new project in Label Studio
   - Set up image classification task
   - Import `results/label_studio_import.json`
   - Model predictions will be pre-filled for quick review

3. **Review and correct:**
   - Review images where model was wrong (check `results/errors.csv`)
   - Correct predictions in Label Studio
   - Export corrected labels

4. **Retrain with corrections:**
   - Move images to correct class folders based on corrections
   - Re-run training pipeline to improve model

This workflow helps you:
- Find and fix labeling errors
- Identify challenging images
- Improve model accuracy iteratively

---

## Docker Usage

### Training (GPU recommended)

```bash
# Build training image
docker build -f docker/Dockerfile.train -t cheese-train .

# Run training
docker run --gpus all -v $(pwd)/images:/app/images -v $(pwd)/models:/app/models cheese-train
```

### Deployment API

```bash
# Build API image
docker build -f docker/Dockerfile.api -t cheese-api .

# Run API
docker run -p 8000:8000 -v $(pwd)/models:/app/models cheese-api

# Test
curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"
```

### Docker Compose (Full Stack)

```bash
docker-compose -f docker/docker-compose.yml up
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify single image |
| `/batch` | POST | Classify multiple images |
| `/health` | GET | Health check |
| `/classes` | GET | List available classes |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@cheese_image.jpg"
```

### Example Response

```json
{
  "class": "white_cubed",
  "confidence": 0.94,
  "probabilities": {
    "white_cubed": 0.94,
    "yellow_cubed": 0.03,
    "shredded": 0.02,
    "nothing": 0.01,
    "cleaning": 0.00
  }
}
```

---

## Configuration

Edit `configs/config.yaml`:

```yaml
model:
  name: efficientnet_b0
  num_classes: 5
  img_size: 224

training:
  epochs: 50
  batch_size: 16
  lr: 0.001
  patience: 10

classes:
  - cleaning
  - nothing
  - shredded
  - white_cubed
  - yellow_cubed
```

---

## Adding New Classes

1. Create folder in `images/new_class/`
2. Add images (50+ recommended)
3. Update `configs/config.yaml` classes list
4. Re-run training

---

## Model Performance

After training, check `models/training_history.json` for:
- Training/validation accuracy per epoch
- Per-class accuracy
- Loss curves

---

## License

All dependencies use permissive licenses (BSD/MIT/Apache) - commercial friendly.
