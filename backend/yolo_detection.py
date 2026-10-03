import cv2
from ultralytics import YOLO

# Load pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# DCSASS video path
video_path = "E:\\DCSASS Dataset\\Fighting\\Fighting002_x264.mp4\\Fighting002_x264_0.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

print("Video opened successfully!")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # YOLO detection
    results = model(frame, verbose=False)

    # Draw detected objects
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()