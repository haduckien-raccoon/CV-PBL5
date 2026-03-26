import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import threading # Thêm thư viện đa luồng

# IMPORT BYTETRACK
from bytetrack.byte_tracker import BYTETracker

# =============================
# CONFIG & CLASSES
# =============================
ENGINE_PATH = "models/yolov8n_best_fp16.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.45
IOU_THRESH = 0.5
CLASSES = ["backpack", "handbag", "laptop", "person", "suitcase"]
COLORS = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 0, 255)]

class ByteTrackArgs:
    track_thresh = 0.45   
    track_buffer = 60      
    match_thresh = 0.8    
    mot20 = False

# =============================
# 1. ĐA LUỒNG CHO CAMERA (Tối ưu I/O)
# =============================
class VideoStream:
    def __init__(self, pipeline):
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        # Khởi chạy một luồng chạy ngầm (daemon) để liên tục đọc frame
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break
            # Dùng lock để tránh xung đột dữ liệu khi luồng chính đang lấy ảnh
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            # Copy frame để luồng chính xử lý mà không ảnh hưởng luồng phụ
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# =============================
# 2. TENSORRT INFERENCE (Giữ nguyên)
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
# 3. GSTREAMER & PRE/POST PROCESS (Đã tối ưu VIC)
# =============================
def gstreamer_pipeline(capture_width=640, capture_height=640, display_width=640, display_height=640, framerate=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv interpolation-method=1 ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"appsink drop=1 max-buffers=1"
    )
    
def preprocess(frame):
    # Khung hình đã là 640x640 từ GStreamer, chỉ cần đổi kênh HWC -> CHW
    img = frame[:, :, ::-1].transpose(2, 0, 1)
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

def get_best_class(track_box, results):
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

def draw_tracked_boxes(frame, tracks, original_results, classes, colors):
    for track in tracks:
        if not track.is_activated: continue
        t_x1, t_y1, t_x2, t_y2 = track.tlbr
        x, y = int(t_x1), int(t_y1)
        w, h = int(t_x2 - t_x1), int(t_y2 - t_y1)
        track_id = track.track_id 
        
        class_id = int(get_best_class([t_x1, t_y1, t_x2, t_y2], original_results))
        class_name = classes[class_id] if class_id < len(classes) else str(class_id)
        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f"ID:{track_id} {class_name}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - label_height - 5), (x + label_width, y), color, -1) 
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# =============================
# 4. MAIN LOOP 
# =============================
def main():
    print("[INFO] Khởi động TensorRT Engine...")
    trt_infer = TensorRTInference(ENGINE_PATH)
    tracker = BYTETracker(ByteTrackArgs())
    
    print("[INFO] Khởi động luồng Camera...")
    # Khởi tạo và BẮT ĐẦU luồng camera
    video_stream = VideoStream(gstreamer_pipeline()).start()
    # Chờ 1 chút để camera kịp lên hình
    time.sleep(1.0) 

    fps_time = time.time()
    frames_count = 0

    print("[INFO] Đang chạy Inference...")
    while True:
        # Lấy frame mới nhất từ luồng phụ (không bị block)
        ret, frame = video_stream.read()
        if frame is None: 
            continue

        # Cắt lấy 3 kênh BGR từ BGRx (rất nhanh)
        if frame.shape[2] == 4:
            frame = np.ascontiguousarray(frame[:, :, :3])

        # Inference
        img = preprocess(frame)
        output = trt_infer.infer(img)
        results = postprocess(output, frame, trt_infer.output_shape, CLASSES)

        # ByteTrack
        dets = []
        for x, y, w, h, conf, cls in results:
            dets.append([x, y, x+w, y+h, conf])
        dets = np.array(dets)

        if len(dets) > 0:
            img_info = [frame.shape[0], frame.shape[1]]
            tracks = tracker.update(dets, img_info, img_info)
        else:
            tracks = []

        # Vẽ và hiển thị
        draw_tracked_boxes(frame, tracks, results, CLASSES, COLORS)

        # Tính toán FPS trung bình mỗi 10 frame để số mượt hơn
        frames_count += 1
        if frames_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
            frames_count = 0
            # Có thể in ra terminal để dễ theo dõi nếu chạy không có màn hình
            # print(f"FPS: {fps:.1f}") 
        
        # Lấy FPS hiện tại (nếu < 10 frame đầu) hoặc dùng FPS đã tính
        current_fps = fps if 'fps' in locals() else 0 
        cv2.putText(frame, f"FPS: {int(current_fps)} | Tracker: ByteTrack", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Robot Tracking Multi-Thread", frame)
        if cv2.waitKey(1) == 27: # Nhấn ESC để thoát
            break

    # Dọn dẹp tài nguyên
    print("[INFO] Đang tắt hệ thống...")
    video_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
