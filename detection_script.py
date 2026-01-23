from ultralytics import YOLO
import os
import glob

# Load your trained model (using available yolov8n.pt)
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found")
    
model = YOLO(model_path)
print("Model loaded successfully!")

# Find a test image from uploads directory
test_image = None
if os.path.exists("uploads"):
    # Try to find a pechay/leaf image
    pechay_images = glob.glob("uploads/*NORMAL*.jpg") + glob.glob("uploads/*ALTERNARIA*.jpg") + glob.glob("uploads/*BLACK-ROT*.jpg")
    if pechay_images:
        test_image = pechay_images[0]
    else:
        # Use any jpg from uploads
        all_images = glob.glob("uploads/*.jpg")
        if all_images:
            test_image = all_images[0]

if not test_image or not os.path.exists(test_image):
    raise FileNotFoundError("No test image found in uploads directory")

print(f"Using test image: {test_image}")

# Predict on a sample image
results = model.predict(test_image, save=True)  # save=True saves annotated image
print("\nPrediction completed! Results saved to 'runs/detect/predict' directory")
# results.show()  # Uncomment to display the annotated image

