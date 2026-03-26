import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# IMPORT SORT (Từ file sort.py tải về trực tiếp)
from sort import Sort

# =============================
# CONFIG & CLASSES
# =============================
ENGINE_PATH = "models/yolov8n_best_fp16.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.45
IOU_THRESH = 0.5
CLASSES = ["backpack", "handbag", "laptop", "person", "suitcase"]
COLORS = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 0, 255)]

# =============================
# TensorRT Class & GStreamer
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

def gstreamer_pipeline():
    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=640,framerate=30/1 ! nvvidconv ! video/x-raw,width=640,height=640,format=BGRx ! appsink drop=1 max-buffers=1")

# =============================
# Logic Xử lý YOLOv8
# =============================
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    return img

def postprocess(output, frame, out_shape, classes):
    h, w = frame.shape[:2]
    num_classes = len(classes)
    output = output.reshape(out_shape).squeeze()
    if output.shape[0] == 8400: output = output.T
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
        x, y, bw, bh = boxes[:,i]
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

# =============================
# Hàm Hỗ trợ SORT: Lấy lại Class
# =============================
def get_best_class(track_box, results):
    # SORT chỉ trả về tọa độ và ID, làm mất Class (Balo, Laptop...).
    # Hàm này so khớp tọa độ của SORT với tọa độ của YOLO để tìm lại Class.
    tx_center = (track_box[0] + track_box[2]) / 2
    ty_center = (track_box[1] + track_box[3]) / 2
    best_class, min_dist = 0, float('inf')
    
    for x, y, w, h, conf, cls in results:
        rx_center = x + w / 2
        ry_center = y + h / 2
        dist = (tx_center - rx_center)**2 + (ty_center - ry_center)**2
        if dist < min_dist:
            min_dist = dist
            best_class = cls
    return best_class

# =============================
# Main Loop (Đã tinh chỉnh cho SORT)
# =============================
def main():
    print("Đang khởi tạo TensorRT YOLOv8...")
    trt_infer = TensorRTInference(ENGINE_PATH)
    
    print("Đang khởi tạo thuật toán SORT...")
    # max_age: Số frame tối đa giữ ID nếu vật thể bị che khuất
    # min_hits: Số frame liên tiếp thấy vật thể mới bắt đầu cấp ID
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue

        if frame.shape[2] == 4:
            frame = frame[:, :, :3].copy()

        # 1. Chạy YOLO
        img = preprocess(frame)
        output = trt_infer.infer(img)
        results = postprocess(output, frame, trt_infer.output_shape, CLASSES)

        # 2. Định dạng Input cho SORT: np.array([[x1, y1, x2, y2, score]])
        dets = []
        for x, y, w, h, conf, cls in results:
            dets.append([x, y, x+w, y+h, conf])
        
        if len(dets) > 0:
            dets = np.array(dets)
        else:
            # RẤT QUAN TRỌNG: Nếu không thấy vật nào, phải truyền mảng rỗng 
            # để SORT cập nhật Kalman Filter (tính thời gian vật mất tích)
            dets = np.empty((0, 5))

        # 3. Cập nhật SORT
        # Output của SORT là: [[x1, y1, x2, y2, track_id], ...]
        tracks = tracker.update(dets)

        # 4. Vẽ Box và ID
        for track in tracks:
            t_x1, t_y1, t_x2, t_y2, track_id = track
            
            x, y = int(t_x1), int(t_y1)
            w, h = int(t_x2 - t_x1), int(t_y2 - t_y1)
            track_id = int(track_id)
            
            # Khôi phục Class Name từ kết quả YOLO gốc
            class_id = int(get_best_class([t_x1, t_y1, t_x2, t_y2], results))
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
            color = COLORS[class_id] if class_id < len(COLORS) else (255, 255, 255)

            # Vẽ lên frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"ID:{track_id} {class_name}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_height - 5), (x + label_width, y), color, -1) 
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 5. Tính FPS và hiển thị
        fps = 1/(time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)} | Tracker: SORT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Robot Tracking", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
