# Install dependencies before running:
# pip install torch torchvision torchaudio
# pip install opencv-python
# pip install ultralytics

import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (YOLOv5 is older, YOLOv8 is easier to use now)
model = YOLO("yolov8n.pt")   # 'n' = nano, fastest & lightest for hackathon demo

# Use webcam (0) or replace with video file path
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, stream=True)
    
    # Draw results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", 
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0,255,0), 2)
    
    cv2.imshow("Video Analysis - YOLOv8", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
