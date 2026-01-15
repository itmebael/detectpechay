"""
Model configuration for Pechay Detection System
"""
import os
from pathlib import Path

# Dataset paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "Pechay Leaf Detection.v1i.yolov9"
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

# Model paths
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov9_pechay.pt"  # Trained model (if available)
CNN_MODEL_PATH = MODELS_DIR / "cnn_pechay.pt"  # Trained CNN model (if available)

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

# Check if dataset exists
DATASET_EXISTS = DATASET_DIR.exists() and DATA_YAML_PATH.exists()

# Class names (from dataset)
CLASS_NAMES = ['Alternaria', 'Blackrot', 'Healthy-Pechay']

# Get data.yaml path as string for use in code
DATA_YAML_STR = str(DATA_YAML_PATH) if DATASET_EXISTS else None
