import cv2
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Define class names (Make sure this order matches exactly with your trained model's class order)
class_names = ['car', 'bike', 'people', 'truck', 'bus', 'taxi']

# Define colors for each class (BGR format for OpenCV)
colors = {
    'car': (0, 255, 0),    # Green
    'bike': (255, 0, 0),   # Blue
    'people': (0, 0, 255), # Red
    'truck': (255, 255, 0),# Cyan
    'bus': (255, 0, 255),  # Magenta
    'taxi': (0, 255, 255)  # Yellow
}

# Get input arguments (image path & model path)
if len(sys.argv) != 3:
    print("Usage: python detect.py <model_path> <image_path>")
    sys.exit(1)

model_path = sys.argv[1]
image_path = sys.argv[2]

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path)
print(f"Model loaded on {device}")

# Print model's class names to verify order
print("\nModel's class names:")
print(model.names)

# Load image
img = cv2.imread(image_path)
if img is None:
    print("Error: Unable to read image. Check the path.")
    sys.exit(1)

# Convert image to RGB for YOLO
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect objects
results = model(img_rgb, conf=0.25)

# Copy image to draw bounding boxes
result_img = img.copy()
class_counts = {}

# Process detections
for r in results:
    for box in r.boxes.data.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
        conf = float(box[4]) if len(box) > 4 else 0.0  # Confidence score
        cls = int(box[5]) if len(box) > 5 else -1  # Class index

        if cls == -1 or cls >= len(model.names):
            continue  # Skip invalid detections

        # Get class name from model's names dictionary
        class_name = model.names[cls]
        color = colors.get(class_name, (0, 255, 0))  # Default green

        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf:.2f}"  # Confidence score

        # Draw label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result_img, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Count detections
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

# Save and display the result
output_path = "result.jpg"
cv2.imwrite(output_path, result_img)

# Show images
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Detection Results")
plt.show()

# Print detection summary
print("\nDetection Summary:")
for class_name, count in class_counts.items():
    print(f"  - {class_name}: {count}")
print(f"Total objects detected: {sum(class_counts.values())}")

print(f"\nResult saved as {output_path}")