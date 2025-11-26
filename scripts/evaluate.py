#!/usr/bin/env python3
"""
Evaluate trained model on images and generate reports.

Usage:
    python scripts/evaluate.py --model models/best_model.pth --images images/ --output results/
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


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
        'predicted_class': class_names[predicted.item()],
        'confidence': confidence.item(),
        'probabilities': {name: float(probs[0, i].item()) for i, name in enumerate(class_names)}
    }


def generate_html_report(results: List[Dict], errors: List[Dict], accuracy: float,
                         class_accuracy: Dict, output_path: Path, images_dir: Path):
    """Generate HTML report with embedded images."""

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cheese Classifier - Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .class-accuracy {{
            margin: 20px 0;
        }}
        .class-bar {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .class-name {{
            width: 150px;
            font-weight: 500;
        }}
        .bar {{
            flex: 1;
            height: 30px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
        }}
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
        }}
        .bar-text {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-weight: bold;
            color: #333;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .results-table th {{
            background: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        .results-table td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        .results-table tr:hover {{
            background: #f5f5f5;
        }}
        .image-cell {{
            width: 120px;
        }}
        .image-cell img {{
            width: 100px;
            height: 100px;
            object-fit: cover;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .image-cell img:hover {{
            transform: scale(1.1);
        }}
        .correct {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .incorrect {{
            color: #f44336;
            font-weight: bold;
        }}
        .confidence {{
            font-family: monospace;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin: 20px 0;
            border-bottom: 2px solid #ddd;
        }}
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            background: #f0f0f0;
            border: none;
            border-radius: 4px 4px 0 0;
            font-size: 16px;
        }}
        .tab.active {{
            background: #4CAF50;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
        }}
        .modal img {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 50px;
        }}
        .close {{
            position: absolute;
            top: 20px;
            right: 40px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧀 Cheese Classifier - Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <div class="stat-card">
                <h3>Total Images</h3>
                <div class="value">{len(results)}</div>
            </div>
            <div class="stat-card">
                <h3>Overall Accuracy</h3>
                <div class="value">{accuracy:.1%}</div>
            </div>
            <div class="stat-card">
                <h3>Correct</h3>
                <div class="value" style="color: #4CAF50;">{len(results) - len(errors)}</div>
            </div>
            <div class="stat-card">
                <h3>Errors</h3>
                <div class="value" style="color: #f44336;">{len(errors)}</div>
            </div>
        </div>

        <h2>Per-Class Accuracy</h2>
        <div class="class-accuracy">
"""

    for cls in sorted(class_accuracy.keys()):
        acc = class_accuracy[cls]
        html_content += f"""
            <div class="class-bar">
                <div class="class-name">{cls}</div>
                <div class="bar">
                    <div class="bar-fill" style="width: {acc*100}%"></div>
                    <div class="bar-text">{acc:.1%}</div>
                </div>
            </div>
"""

    html_content += """
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('errors')">Errors Only</button>
            <button class="tab" onclick="showTab('all')">All Results</button>
        </div>

        <div id="errors" class="tab-content active">
            <h2>Incorrect Predictions ({} errors)</h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Image</th>
                        <th>Filename</th>
                        <th>Actual</th>
                        <th>Predicted</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
""".format(len(errors))

    for error in errors:
        img_path = Path(error['image_path'])
        rel_path = img_path.relative_to(images_dir) if img_path.is_absolute() else img_path
        html_content += f"""
                    <tr>
                        <td class="image-cell">
                            <img src="{rel_path}" alt="{error['filename']}" onclick="showModal(this.src)">
                        </td>
                        <td>{error['filename']}</td>
                        <td><span class="correct">{error['actual_class']}</span></td>
                        <td><span class="incorrect">{error['predicted_class']}</span></td>
                        <td class="confidence">{error['confidence']:.2%}</td>
                    </tr>
"""

    html_content += """
                </tbody>
            </table>
        </div>

        <div id="all" class="tab-content">
            <h2>All Predictions</h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Image</th>
                        <th>Filename</th>
                        <th>Actual</th>
                        <th>Predicted</th>
                        <th>Confidence</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""

    for result in results:
        img_path = Path(result['image_path'])
        rel_path = img_path.relative_to(images_dir) if img_path.is_absolute() else img_path
        status_class = 'correct' if result['correct'] else 'incorrect'
        status_text = '✓ Correct' if result['correct'] else '✗ Wrong'

        html_content += f"""
                    <tr>
                        <td class="image-cell">
                            <img src="{rel_path}" alt="{result['filename']}" onclick="showModal(this.src)">
                        </td>
                        <td>{result['filename']}</td>
                        <td>{result['actual_class']}</td>
                        <td>{result['predicted_class']}</td>
                        <td class="confidence">{result['confidence']:.2%}</td>
                        <td class="{status_class}">{status_text}</td>
                    </tr>
"""

    html_content += """
                </tbody>
            </table>
        </div>
    </div>

    <!-- Image Modal -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img id="modalImage" src="">
    </div>

    <script>
        function showTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(el => {
                el.classList.remove('active');
            });
            document.querySelectorAll('.tab').forEach(el => {
                el.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        function showModal(src) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def evaluate_images(model_path: str, images_dir: str, output_dir: str):
    """Evaluate model on all images and generate reports."""

    # Setup
    device = get_device()
    print(f"Using device: {device}")

    model, class_names, img_size = load_model(model_path, device)
    transform = get_transform(img_size)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model classes: {class_names}")
    print(f"Evaluating images from: {images_dir}")

    # Find all images organized by class folders
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
    class_dirs = [d for d in images_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not class_dirs:
        print(f"Error: No class folders found in {images_dir}")
        return

    # Collect all images with ground truth labels
    image_data = []
    for class_dir in class_dirs:
        actual_class = class_dir.name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]

        for img_path in images:
            image_data.append({
                'path': img_path,
                'actual_class': actual_class,
                'filename': img_path.name,
                'relative_path': str(img_path.relative_to(images_dir))
            })

    if not image_data:
        print("Error: No images found")
        return

    print(f"\nFound {len(image_data)} images across {len(class_dirs)} classes")
    print(f"Classes: {sorted([d.name for d in class_dirs])}")

    # Run predictions
    results = []
    errors = []
    correct_count = 0
    class_stats = {cls: {'total': 0, 'correct': 0} for cls in class_names}

    print("\nRunning predictions...")
    for item in tqdm(image_data, desc="Evaluating"):
        try:
            image = Image.open(item['path']).convert('RGB')
            prediction = predict_image(model, image, transform, class_names, device)

            actual = item['actual_class']
            predicted = prediction['predicted_class']
            is_correct = actual == predicted

            if is_correct:
                correct_count += 1

            # Update class stats
            if actual in class_stats:
                class_stats[actual]['total'] += 1
                if is_correct:
                    class_stats[actual]['correct'] += 1

            # Build result record
            result = {
                'image_path': str(item['path']),
                'filename': item['filename'],
                'relative_path': item['relative_path'],
                'actual_class': actual,
                'predicted_class': predicted,
                'confidence': prediction['confidence'],
                'correct': is_correct,
                **{f'prob_{cls}': prediction['probabilities'][cls] for cls in class_names}
            }

            results.append(result)

            if not is_correct:
                errors.append(result)

        except Exception as e:
            print(f"\nError processing {item['path']}: {e}")
            continue

    # Calculate statistics
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0

    class_accuracy = {}
    for cls, stats in class_stats.items():
        if stats['total'] > 0:
            class_accuracy[cls] = stats['correct'] / stats['total']
        else:
            class_accuracy[cls] = 0.0

    # Save predictions CSV
    predictions_csv = output_dir / 'predictions.csv'
    with open(predictions_csv, 'w', newline='') as f:
        if results:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\n✓ Saved all predictions to: {predictions_csv}")

    # Save errors CSV
    errors_csv = output_dir / 'errors.csv'
    with open(errors_csv, 'w', newline='') as f:
        if errors:
            fieldnames = list(errors[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(errors)

    print(f"✓ Saved errors to: {errors_csv} ({len(errors)} errors)")

    # Save Label Studio JSON
    label_studio_tasks = []
    for result in results:
        task = {
            'data': {
                'image': f"/data/local-files/?d={result['relative_path']}"
            },
            'predictions': [{
                'result': [{
                    'value': {
                        'choices': [result['predicted_class']]
                    },
                    'score': result['confidence'],
                    'from_name': 'choice',
                    'to_name': 'image',
                    'type': 'choices'
                }],
                'score': result['confidence'],
                'model_version': Path(model_path).stem
            }],
            'meta': {
                'filename': result['filename'],
                'actual_class': result['actual_class'],
                'predicted_class': result['predicted_class'],
                'correct': result['correct']
            }
        }
        label_studio_tasks.append(task)

    label_studio_json = output_dir / 'label_studio_import.json'
    with open(label_studio_json, 'w') as f:
        json.dump(label_studio_tasks, f, indent=2)

    print(f"✓ Saved Label Studio import to: {label_studio_json}")

    # Generate HTML report with images
    html_report = output_dir / 'report.html'
    generate_html_report(results, errors, accuracy, class_accuracy, html_report, images_dir)
    print(f"✓ Saved HTML report with images to: {html_report}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'images_directory': str(images_dir),
        'total_images': total,
        'correct_predictions': correct_count,
        'incorrect_predictions': len(errors),
        'overall_accuracy': accuracy,
        'class_accuracy': class_accuracy,
        'class_statistics': class_stats,
        'classes': class_names
    }

    summary_json = output_dir / 'summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary to: {summary_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total images: {total}")
    print(f"Correct: {correct_count} ({accuracy:.1%})")
    print(f"Incorrect: {len(errors)} ({(1-accuracy):.1%})")
    print(f"\nPer-class accuracy:")
    for cls in sorted(class_accuracy.keys()):
        acc = class_accuracy[cls]
        stats = class_stats[cls]
        print(f"  {cls:20s}: {acc:.1%} ({stats['correct']}/{stats['total']})")
    print("=" * 60)

    if errors:
        print(f"\nReview errors in: {errors_csv}")
        print(f"Import to Label Studio: {label_studio_json}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model and generate reports')
    parser.add_argument('--model', default='models/best_model.pth', help='Path to trained model')
    parser.add_argument('--images', default='images', help='Directory with class folders')
    parser.add_argument('--output', default='results', help='Output directory for reports')

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Train a model first: python scripts/train.py")
        return

    evaluate_images(args.model, args.images, args.output)


if __name__ == '__main__':
    main()
