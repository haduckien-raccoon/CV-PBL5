import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# =============================
# CONFIG
# =============================
ENGINE_PATH = "models/yolov8n_best_fp16.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.4
IOU_THRESH = 0.5

# =============================
# TensorRT Init
# =============================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Chuyển shape thành tuple tĩnh để dễ tái sử dụng
input_shape = tuple(engine.get_binding_shape(0))
output_shape = tuple(engine.get_binding_shape(1))

input_size = abs(trt.volume(input_shape))
output_size = abs(trt.volume(output_shape))

d_input = cuda.mem_alloc(input_size * 4)
d_output = cuda.mem_alloc(output_size * 4)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
host_output = np.empty(output_size, dtype=np.float32)

print("Engine input shape:", input_shape)
print("Engine output shape:", output_shape)

# =============================
# Camera pipeline (Tối ưu GPU)
# =============================
def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw,width=640,height=640,format=BGRx ! "
        "appsink drop=1"
    )

# =============================
# Logic Xử lý
# =============================
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1] # BGR to RGB
    img = img.transpose(2, 0, 1) # HWC to CHW
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    return img

def nms(boxes, scores):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    if len(idxs) == 0:
        return []
    return idxs.flatten()

def decode(output, frame, out_shape):
    h, w = frame.shape[:2]
    
    # Fix triệt để lỗi loạn box do transpose memory
    output = output.reshape(out_shape).squeeze()
    if output.shape[0] == 8400:
        output = output.T

    boxes = output[0:4, :]
    scores = output[4:, :]

    class_ids = np.argmax(scores, axis=0)
    confidences = np.max(scores, axis=0)
    mask = confidences > CONF_THRESH

    boxes = boxes[:, mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    detections = []
    scores_list = []

    for i in range(boxes.shape[1]):
        x, y, bw, bh = boxes[:,i]
        
        # Scale back to frame size
        x1 = int((x - bw/2) * w / INPUT_SIZE)
        y1 = int((y - bh/2) * h / INPUT_SIZE)
        x2 = int((x + bw/2) * w / INPUT_SIZE)
        y2 = int((y + bh/2) * h / INPUT_SIZE)

        detections.append([x1, y1, x2-x1, y2-y1])
        scores_list.append(float(confidences[i]))

    idxs = nms(detections, scores_list)
    results = [(detections[i][0], detections[i][1], detections[i][2], detections[i][3], scores_list[i], class_ids[i]) for i in idxs]
    return results

# =============================
# Inference loop
# =============================
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Camera failed")
    exit()

fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: continue

    # Chuyển BGRx (4 kênh) từ nvvidconv sang BGR (3 kênh) tốc độ cao
    if frame.shape[2] == 4:
        frame = frame[:, :, :3].copy()

    img = preprocess(frame)

    cuda.memcpy_htod_async(d_input, img, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, d_output, stream)
    stream.synchronize()

    # Truyền output_shape vào để decode không bị nhầm
    results = decode(host_output, frame, output_shape)

    for x, y, w, h, conf, cls in results:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls}:{conf:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    fps = 1/(time.time() - fps_time)
    fps_time = time.time()

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 TensorRT", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
