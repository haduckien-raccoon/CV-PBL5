import cv2

pipeline = (
"nvarguscamerasrc ! "
"video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
"nvvidconv ! video/x-raw,format=BGRx ! "
"videoconvert ! video/x-raw,format=BGR ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera open failed")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Frame grab failed")
        break

    cv2.imshow("CSI Camera", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
