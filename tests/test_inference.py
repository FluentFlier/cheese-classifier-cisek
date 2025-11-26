#!/usr/bin/env python3
"""
Basic tests for cheese classifier.

Usage:
    pytest tests/test_inference.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model_exists():
    """Check if model file exists."""
    model_path = Path("models/best_model.pth")
    # Skip if no model trained yet
    if not model_path.exists():
        import pytest
        pytest.skip("No model trained yet")
    assert model_path.exists()


def test_load_model():
    """Test model loading."""
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        import pytest
        pytest.skip("No model trained yet")
    
    import torch
    checkpoint = torch.load(model_path, map_location='cpu')
    
    assert 'model_state_dict' in checkpoint
    assert 'class_names' in checkpoint
    assert len(checkpoint['class_names']) > 0


def test_config_exists():
    """Check if config file exists."""
    config_path = Path("configs/config.yaml")
    assert config_path.exists()


def test_config_valid():
    """Test config is valid YAML."""
    import yaml
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    assert 'model' in config
    assert 'training' in config
    assert 'classes' in config


def test_scripts_exist():
    """Check all scripts exist."""
    scripts = [
        "scripts/split_data.py",
        "scripts/train.py",
        "scripts/inference.py",
        "app.py"
    ]
    for script in scripts:
        assert Path(script).exists(), f"Missing: {script}"
