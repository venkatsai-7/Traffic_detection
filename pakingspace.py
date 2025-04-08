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

# Get input arguments (model path & video path)
if len(sys.argv) != 3:
    print("Usage: python detect_video.py <model_path> <video_path>")
    sys.exit(1)

model_path = sys.argv[1]
video_path = sys.argv[2]

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path)
print(f"Model loaded on {device}")

# Print model's class names to verify order
print("\nModel's class names:")
print(model.names)

# Open video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Unable to open video. Check the path.")
    sys.exit(1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output video writer
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Initialize frame counter and class counts
frame_count = 0
total_class_counts = {}

print("\nProcessing video...")

# Set confidence threshold
CONF_THRESHOLD = 0.45  # Increased confidence threshold for better accuracy

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"\rProcessing frame {frame_count}", end="")

    # Convert frame to RGB for YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects with higher confidence threshold
    results = model(frame_rgb, conf=CONF_THRESHOLD, iou=0.45)  # Added IOU threshold

    # Process detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls >= len(model.names):
                continue  # Skip invalid detections

            # Get class name from model's names dictionary
            class_name = model.names[cls]
            color = colors.get(class_name, (0, 255, 0))  # Default green

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name}: {conf:.2f}"

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Count detections
            total_class_counts[class_name] = total_class_counts.get(class_name, 0) + 1

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow('Video Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# Print final detection summary
print("\n\nFinal Detection Summary:")
for class_name, count in total_class_counts.items():
    print(f"  - {class_name}: {count}")
print(f"Total objects detected: {sum(total_class_counts.values())}")
print(f"Total frames processed: {frame_count}")
print(f"\nResult saved as {output_path}")
