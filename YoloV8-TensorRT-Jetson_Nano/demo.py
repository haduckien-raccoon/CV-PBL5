import cv2
import time
from ultralytics import YOLO

print("Loading YOLOv8 model...")
model = YOLO("models/yolov8n_best.engine")   # hoặc đường dẫn model của bạn

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

print("Opening CSI camera...")

cap = cv2.VideoCapture(
    gstreamer_pipeline(),
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    print("Camera failed")
    exit()

fps_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame grab failed")
        break

    # YOLO inference
    results = model(frame)

    annotated = results[0].plot()

    # FPS calculation
    fps = 1 / (time.time() - fps_time)
    fps_time = time.time()

    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv8 Jetson Nano", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
