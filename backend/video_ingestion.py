import cv2

video_path = r"E:\DCSASS Dataset\Robbery\Robbery001_x264.mp4\Robbery001_x264_0.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Video opened successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    cv2.imshow("DCSASS CCTV Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()