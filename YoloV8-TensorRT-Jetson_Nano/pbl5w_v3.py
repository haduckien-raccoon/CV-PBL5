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
CONF_THRESH = 0.45
IOU_THRESH = 0.5

CLASSES = ["backpack", "handbag", "laptop", "person", "suitcase"]
COLORS = [
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (0, 0, 255)
]

# =============================
# TensorRT Class
# =============================
class TensorRTInference:
    def __init__(self, engine_path):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_binding_idx = 0
        self.output_binding_idx = 1
        self.input_shape = tuple(self.engine.get_binding_shape(self.input_binding_idx))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_binding_idx))
        self.input_volume = abs(trt.volume(self.input_shape))
        self.output_volume = abs(trt.volume(self.output_shape))
        self.d_input = cuda.mem_alloc(self.input_volume * 4)
        self.d_output = cuda.mem_alloc(self.output_volume * 4)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
        self.host_output = np.empty(self.output_volume, dtype=np.float32)

    def infer(self, input_data):
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.host_output

# =============================
# Preprocess / Postprocess
# =============================
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0
    return img

def postprocess(output, frame, out_shape, classes):
    h, w = frame.shape[:2]
    num_classes = len(classes)
    output = output.reshape(out_shape).squeeze()
    if output.shape[0] == 8400:
        output = output.T
    boxes = output[0:4, :]
    scores = output[4:(4+num_classes), :]
    class_ids = np.argmax(scores, axis=0)
    confidences = np.max(scores, axis=0)
    mask = confidences > CONF_THRESH
    boxes = boxes[:, mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    detections, scores_list = [], []
    for i in range(boxes.shape[1]):
        x, y, bw, bh = boxes[:, i]
        x1 = max(0, int((x - bw/2) * w / INPUT_SIZE))
        y1 = max(0, int((y - bh/2) * h / INPUT_SIZE))
        width = int(bw * w / INPUT_SIZE)
        height = int(bh * h / INPUT_SIZE)
        detections.append([x1, y1, width, height])
        scores_list.append(float(confidences[i]))
    idxs = cv2.dnn.NMSBoxes(detections, scores_list, CONF_THRESH, IOU_THRESH)
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            results.append((detections[i][0], detections[i][1], detections[i][2], detections[i][3], scores_list[i], class_ids[i]))
    return results

def draw_boxes(frame, results, classes, colors):
    for x, y, w, h, conf, cls in results:
        class_id = int(cls)
        class_name = classes[class_id] if class_id < len(classes) else str(class_id)
        color = colors[class_id] if class_id < len(colors) else (255,255,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"{class_name} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y-label_height-5), (x+label_width, y), color, -1)
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

# =============================
# Main
# =============================
def main():
    print("Khởi tạo TensorRT Engine...")
    trt_infer = TensorRTInference(ENGINE_PATH)
    
    print("Mở webcam /dev/video1...")
    cap = cv2.VideoCapture(1)  # /dev/video1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # giảm lag

    if not cap.isOpened():
        print("❌ Cannot open camera /dev/video1")
        return

    fps_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: 
            continue

        img = preprocess(frame)
        output = trt_infer.infer(img)
        results = postprocess(output, frame, trt_infer.output_shape, CLASSES)
        draw_boxes(frame, results, CLASSES, COLORS)

        fps = 1/(time.time()-fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("YOLOv8 TensorRT USB Cam", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
