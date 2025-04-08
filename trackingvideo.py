import cv2
import torch
import numpy as np
import sys
import math
from ultralytics import YOLO
import cvzone
from sort import *

# Define class names for vehicle detection
class_names = ['car', 'bike', 'truck', 'bus', 'taxi']

# Define colors for each class (BGR format for OpenCV)
colors = {
    'car': (0, 255, 0),    # Green
    'bike': (255, 0, 0),   # Blue
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

# Set confidence threshold
CONF_THRESHOLD = 0.35  # Balanced threshold for speed and accuracy

# Initialize tracker with optimized parameters
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.35)  # Faster tracking

# Define detection zones as rectangles with optimized sizes
# Format: [x1, y1, x2, y2] - top-left and bottom-right corners
zones = {
    'heavy': {
        'rect': [100, 250, 900, 400],    # Bottom zone (red) - wider for better detection
        'count': 0,
        'vehicles': []
    },
    'medium': {
        'rect': [100, 400, 900, 550],    # Middle zone (yellow)
        'count': 0,
        'vehicles': []
    },
    'normal': {
        'rect': [100, 550, 900, 700],    # Top zone (green)
        'count': 0,
        'vehicles': []
    }
}

# Traffic status colors
traffic_colors = {
    'normal': (0, 255, 0),      # Green
    'medium': (0, 255, 255),    # Yellow
    'heavy': (0, 0, 255)        # Red
}

# Traffic thresholds
TIME_THRESHOLDS = {
    'normal': 1,     # Less than 1 second = normal traffic (will show as heavy)
    'medium': 3,     # 1-3 seconds = medium congestion
    'heavy': 5       # More than 5 seconds = heavy congestion
}

# Zone-specific thresholds
ZONE_THRESHOLDS = {
    'start_to_middle': {
        'normal': 0.5,    # Less than 0.5 seconds (will show as heavy)
        'medium': 1.5,    # 0.5-1.5 seconds
        'heavy': 2.5      # More than 2.5 seconds
    },
    'middle_to_end': {
        'normal': 0.3,    # Less than 0.3 seconds (will show as heavy)
        'medium': 1.0,    # 0.3-1.0 seconds
        'heavy': 1.5      # More than 1.5 seconds
    }
}

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

# Initialize frame counter
frame_count = 0

print("\nProcessing video...")

def get_traffic_status(zones):
    """Determine traffic status based on vehicle counts in different zones"""
    if not zones:
        return 'normal'
    
    # Calculate total vehicles in each zone
    total_vehicles = sum(zone['count'] for zone in zones.values())
    
    # Get counts for each zone
    normal_zone = zones['normal']['count']
    medium_zone = zones['medium']['count']
    heavy_zone = zones['heavy']['count']
    
    # Determine status based on vehicle distribution
    if total_vehicles == 0:
        return 'normal'
    
    # Calculate ratios of vehicles in each zone
    normal_ratio = normal_zone / total_vehicles if total_vehicles > 0 else 0
    medium_ratio = medium_zone / total_vehicles if total_vehicles > 0 else 0
    heavy_ratio = heavy_zone / total_vehicles if total_vehicles > 0 else 0
    
    # Optimized traffic status conditions
    if heavy_zone >= 2 or (heavy_ratio > 0.25 and medium_zone >= 2):  # More sensitive heavy detection
        return 'heavy'
    elif medium_zone >= 3 or (medium_ratio > 0.3 and normal_zone >= 2):  # More sensitive medium detection
        return 'medium'
    else:
        return 'normal'

def is_point_in_rect(point, rect):
    """Check if a point is inside a rectangle with optimized bounds"""
    x, y = point
    x1, y1, x2, y2 = rect
    buffer = 5
    return (x1 - buffer) <= x <= (x2 + buffer) and (y1 - buffer) <= y <= (y2 + buffer)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"\rProcessing frame {frame_count}", end="")

    # Resize frame for faster processing
    frame = cv2.resize(frame, (width, height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects with optimized parameters
    results = model(frame_rgb, conf=CONF_THRESHOLD, iou=0.35, verbose=False)

    # Initialize detections array for tracking
    detections = np.empty((0, 5))

    # Process detections with optimized loop
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls >= len(model.names):
                continue

            class_name = model.names[cls]
            
            if class_name in class_names and conf >= CONF_THRESHOLD:
                currentArray = np.array([[x1, y1, x2, y2, conf]])
                detections = np.vstack((detections, currentArray))

                # Draw bounding box with class name and confidence
                w, h = x2 - x1, y2 - y1
                color = colors.get(class_name, (255, 255, 255))
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=color)
                 
                # Display class name with confidence
                label = f"{class_name} {conf:.2f}"
                cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1)),
                                 scale=1, thickness=2, offset=10)

    # Update tracker with optimized parameters
    resultsTracker = tracker.update(detections)

    # Reset zone counts efficiently
    for zone in zones.values():
        zone['count'] = 0
        zone['vehicles'].clear()  # More efficient than reassignment

    # Draw zones and process tracked objects with optimized loop
    for zone_name, zone_data in zones.items():
        rect = zone_data['rect']
        # Draw zone rectangle with semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), 
                     traffic_colors[zone_name], -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), 
                     traffic_colors[zone_name], 2)

        # Process tracked objects for this zone
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Calculate center point
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Check if vehicle is in the zone with optimized detection
            if is_point_in_rect((cx, cy), rect):
                if id not in zone_data['vehicles']:
                    zone_data['vehicles'].append(id)
                    zone_data['count'] += 1
                    # Highlight zone when vehicle enters
                    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), 
                                (0, 255, 0), 2)

    # Get current traffic status
    traffic_status = get_traffic_status(zones)

    # Display traffic information
    info_panel = np.zeros((250, 300, 3), dtype=np.uint8)
    
    # Display counts for each zone
    y_offset = 30
    for zone_name, zone_data in zones.items():
        cv2.putText(info_panel, f'{zone_name.capitalize()}: {zone_data["count"]}', 
                   (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, 
                   traffic_colors[zone_name], 2)
        y_offset += 30

    # Display total vehicles and traffic status
    total_vehicles = sum(zone['count'] for zone in zones.values())
    cv2.putText(info_panel, f'Total Vehicles: {total_vehicles}', 
                (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
    
    status_text = f'Traffic Status: {traffic_status.upper()}'
    if traffic_status == 'heavy':
        status_text += ' (HEAVY TRAFFIC)'
    elif traffic_status == 'medium':
        status_text += ' (MODERATE TRAFFIC)'
    cv2.putText(info_panel, status_text, 
                (10, y_offset + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, 
                traffic_colors[traffic_status], 2)
    
    # Add info panel to frame
    frame[10:260, 10:310] = info_panel

    # Write frame to output video
    out.write(frame)

    # Display frame
    cv2.imshow('Traffic Density Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# Print final detection summary
print("\n\nFinal Detection Summary:")
for zone_name, zone_data in zones.items():
    print(f"{zone_name.capitalize()} zone vehicles: {zone_data['count']}")
print(f"Total vehicles: {sum(zone['count'] for zone in zones.values())}")
print(f"Final traffic status: {get_traffic_status(zones).upper()}")
print(f"Total frames processed: {frame_count}")
print(f"\nResult saved as {output_path}")