# Quick Start - Using YOLOv9 Dataset for Detection

## Your Dataset

✅ Your YOLOv9 dataset is configured and ready to use!

**Dataset Location:** `Pechay Leaf Detection.v1i.yolov9/`
- 3 Classes: Alternaria, Blackrot, Healthy-Pechay
- 4,597 training images
- 60 validation images
- 29 test images

## How Detection Works Now

The code has been updated to:
1. ✅ Load class names from your dataset (`data.yaml`)
2. ✅ Use the correct dataset structure
3. ✅ Automatically detect if a trained model exists
4. ✅ Use pre-trained YOLOv9 if no trained model is found

## Two Options

### Option 1: Train Custom Model (Best Results) ⭐

Train a YOLOv9 model on your dataset:

```bash
python train_yolo.py
```

This will:
- Fine-tune YOLOv9 on your Pechay dataset
- Save trained model to `models/yolov9_pechay.pt`
- Take time (depends on GPU/CPU)

### Option 2: Use Pre-trained Model (Quick Start)

The system will automatically use a pre-trained YOLOv9 model if no trained model is found. Accuracy will be lower than a custom-trained model.

## Current Status

✅ **Dataset configured** - Class names loaded from your dataset
✅ **Code updated** - Ready to use YOLOv9 for detection
⚠️ **Model needed** - Either train one or use pre-trained

## Next Steps

1. **For best results**: Train the model (`python train_yolo.py`)
2. **For quick testing**: Run the app and it will use pre-trained YOLOv9
3. **Detection will use your classes**: Alternaria, Blackrot, Healthy-Pechay

The detection system is now configured to use your YOLOv9 dataset! 🎉
















