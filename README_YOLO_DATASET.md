# Using YOLOv9 Dataset for Detection

## Dataset Structure

Your YOLOv9 dataset is located at: `Pechay Leaf Detection.v1i.yolov9/`

### Dataset Contents:
- **Classes**: 3 classes (Alternaria, Blackrot, Healthy-Pechay)
- **Training images**: 4,597 images
- **Validation images**: 60 images  
- **Test images**: 29 images
- **Configuration**: `data.yaml` with dataset paths and class names

## Two Options for Detection

### Option 1: Train Your Own Model (Recommended for Best Results)

1. **Install dependencies:**
   ```bash
   pip install ultralytics torch torchvision
   ```

2. **Train the model:**
   ```bash
   python train_yolo.py
   ```
   
   This will:
   - Load the pre-trained YOLOv9 model
   - Fine-tune it on your Pechay dataset
   - Save the trained model to `models/yolov9_pechay.pt`

3. **The trained model will be automatically used** by the detection system

### Option 2: Use Pre-trained YOLOv9 (Quick Start)

The code is configured to use a pre-trained YOLOv9 model if no trained model is found. However, this will have lower accuracy since it hasn't been trained on your specific dataset.

## Current Implementation

The code has been updated to:
1. ✅ Load class names from your dataset (`data.yaml`)
2. ✅ Use the dataset structure for model configuration
3. ✅ Automatically detect if a trained model exists
4. ✅ Fall back to pre-trained model if needed

## Next Steps

1. **Train the model** (recommended):
   ```bash
   python train_yolo.py
   ```
   
2. **Or use pre-trained** (for testing):
   - The system will use pre-trained YOLOv9
   - Accuracy will be lower than a trained model
   
3. **The detection will automatically use your dataset classes:**
   - Alternaria
   - Blackrot  
   - Healthy-Pechay

## Model Files

- Trained model should be saved at: `models/yolov9_pechay.pt`
- If not found, the system uses: `yolov9c.pt` (pre-trained)

## Training Notes

- Training takes time (depends on GPU/CPU)
- Recommended: Use GPU for faster training
- Training parameters are in `train_yolo.py`
- Adjust epochs, batch size, etc. as needed



















