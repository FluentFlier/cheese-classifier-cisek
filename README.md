# Cheese Inspection System

Binary classification: **cheese** vs **no_cheese**

Uses your COCO bounding box annotations - any image with a box = cheese, no boxes = no_cheese.

## Licensing
All dependencies use permissive licenses (BSD/MIT/Apache) - fully commercial-friendly.

---

## Quick Start (One Command)

```bash
# Activate your environment
source venv/bin/activate

# Run everything
python run_all.py --coco path/to/result.json --images path/to/images/
```

That's it. Model saves to `models/best_model.pth`.

---

## Step-by-Step Setup

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Export from Label Studio

Export your project as **COCO format** (with images). You'll get:
```
export/
├── result.json      # COCO annotations
└── images/          # Your labeled images
```

### 3. Train

**Option A - One command:**
```bash
python run_all.py --coco export/result.json --images export/images/
```

**Option B - Step by step:**
```bash
# Prepare data
python scripts/prepare_from_coco.py \
    --coco export/result.json \
    --images export/images/ \
    --output data/

# Train
python scripts/train.py --data data/ --output models/ --epochs 30
```

### 4. Test

```bash
# Single image
python scripts/inference.py --model models/best_model.pth --image test.jpg

# Folder of images
python scripts/inference.py --model models/best_model.pth --dir test_images/ --output results.json

# Live camera
python scripts/inference.py --model models/best_model.pth --video 0
```

**Video file:**
```bash
python scripts/inference.py --model models/best_model.pth --video recording.mp4
```

---

## Training Tips for Better Accuracy

### Data Quality
- **Minimum**: 50 images per class
- **Recommended**: 200+ images per class
- Include variety in lighting, angles, cheese quantity
- Balance your classes (roughly equal samples per class)

### Data Augmentation
The training script includes aggressive augmentation:
- Random crops, flips, rotations
- Color jitter (brightness, contrast, saturation)
- Random erasing

This helps the model generalize better with limited data.

### Hyperparameters
```bash
# For small datasets (<100 images)
python scripts/train.py --epochs 100 --batch-size 8 --lr 0.0005

# For larger datasets (500+ images)
python scripts/train.py --epochs 50 --batch-size 32 --lr 0.001
```

### If Accuracy is Poor
1. **Check class balance** - Ensure roughly equal samples per class
2. **More data** - Label more images, especially failure cases
3. **Review labels** - Check for mislabeled images
4. **Lower learning rate** - Try `--lr 0.0001`
5. **More epochs** - Try `--epochs 100` with higher patience

---

## Adding More Classes

To add yellow_cubed, shredded, cleaning, etc:

1. Update `configs/label_studio_config.xml`:
```xml
<View>
  <Image name="image" value="$image"/>
  <Choices name="choice" toName="image" choice="single" showInline="true">
    <Choice value="white_cubed"/>
    <Choice value="yellow_cubed"/>
    <Choice value="shredded"/>
    <Choice value="nothing"/>
    <Choice value="cleaning"/>
  </Choices>
</View>
```

2. Update `CLASS_NAMES` in `scripts/train.py`:
```python
CLASS_NAMES = ['cleaning', 'nothing', 'shredded', 'white_cubed', 'yellow_cubed']
```
Note: Names must be alphabetically sorted to match ImageFolder behavior.

3. Re-label and retrain.

---

## Project Structure

```
cheese-inspection/
├── run_all.py                    # One-command training
├── scripts/
│   ├── prepare_from_coco.py      # Parse COCO export → dataset
│   ├── train.py                  # Training script
│   └── inference.py              # Inference script
├── data/                         # Generated: train/val splits
├── models/                       # Generated: saved models
├── requirements.txt
└── README.md
```

---

## Deployment

The trained model can be deployed:

**Python/Flask API:**
```python
from scripts.inference import load_model, predict_image, get_inference_transform
from PIL import Image
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, class_names = load_model('models/best_model.pth', device)
transform = get_inference_transform()

# In your API endpoint:
image = Image.open(uploaded_file).convert('RGB')
result = predict_image(model, image, transform, class_names, device)
```

**Edge deployment (Jetson, Raspberry Pi):**
- Model size: ~16MB
- Works on CPU (slower) or GPU
- For faster inference, consider ONNX export or TensorRT

---

## Troubleshooting

**CUDA out of memory:**
- Reduce `--batch-size` (try 8 or 4)

**Model not improving:**
- Check data quality and labels
- Try lower learning rate
- Ensure classes are balanced

**Slow training:**
- Use GPU if available
- Reduce `--img-size` to 192 or 160

**Import errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
