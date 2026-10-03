import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# DCSASS video
video_path = "E:\\DCSASS Dataset\\Fighting\\Fighting014_x264.mp4\\Fighting014_x264_0.mp4"

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

    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw detections and tracking IDs
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 + ByteTrack", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()