from ultralytics import YOLO

# Load your trained model
model = YOLO("Pechay Leaf Detection.v1i.yolov9")
print("Model loaded successfully!")

# Predict on a sample image
results = model.predict("test_leaf.jpg", save=True)  # save=True saves annotated image
results.show()  # Display the annotated image

