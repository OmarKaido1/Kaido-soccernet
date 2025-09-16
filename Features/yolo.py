import cv2
import numpy as np
import os

# YOLOv3 files
yolo_cfg = "yolov3.cfg"
yolo_weights = "yolov3.weights"
coco_names = "coco.names"

# Load YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load labels
with open(coco_names, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Input video
video_path = "input.mp4"
output_dir = "tracked_balls"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

tracker = None
init_once = False
frame_id = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    height, width = frame.shape[:2]

    # If tracker not initialized → detect with YOLO
    if tracker is None:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences = [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if classes[class_id] == "sports ball" and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        if boxes:
            # pick highest confidence ball
            best_idx = np.argmax(confidences)
            x, y, w, h = boxes[best_idx]

            # initialize tracker with this box
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            init_once = True

    else:
        # Update tracker
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]

            # Expand crop region slightly
            pad_x, pad_y = int(0.2 * w), int(0.2 * h)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)

            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                save_path = os.path.join(output_dir, f"ball_{frame_id:05d}.jpg")
                cv2.imwrite(save_path, cropped)
                saved_count += 1

            # Draw box for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, "Ball", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        else:
            # If tracking failed → reset and use YOLO again
            tracker = None

    # Show video with tracking
    cv2.imshow("Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Saved {saved_count} cropped ball images in '{output_dir}'")
