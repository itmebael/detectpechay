"""
Script to train YOLOv9 model on Pechay dataset
Run this to train your custom YOLOv9 model
"""
from ultralytics import YOLO
from pathlib import Path
import os

# Dataset path
DATA_YAML = "Pechay Leaf Detection.v1i.yolov9/data.yaml"

# Check if dataset exists
if not os.path.exists(DATA_YAML):
    print(f"Error: Dataset not found at {DATA_YAML}")
    print("Please ensure the dataset folder exists.")
    exit(1)

# Initialize YOLOv9 model
print("Loading YOLOv9 pre-trained model...")
model = YOLO('yolov9c.pt')  # Use YOLOv9c (medium size)

# Train the model
print("Starting training...")
print(f"Dataset config: {DATA_YAML}")

results = model.train(
    data=DATA_YAML,
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size
    batch=16,  # Batch size (adjust based on your GPU memory)
    name='yolov9_pechay',  # Project name
    project='runs/detect',  # Project directory
    save=True,
    save_period=10,  # Save checkpoint every 10 epochs
    val=True,  # Validate during training
    plots=True,  # Generate plots
    verbose=True
)

print("Training completed!")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")

# Copy best model to models directory
import shutil
models_dir = Path('models')
models_dir.mkdir(exist_ok=True)
best_model = Path(results.save_dir) / 'weights' / 'best.pt'
if best_model.exists():
    shutil.copy(best_model, models_dir / 'yolov9_pechay.pt')
    print(f"Model copied to: {models_dir / 'yolov9_pechay.pt'}")










