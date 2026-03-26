import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# IMPORT BYTETRACK (Từ thư mục bạn vừa tạo)
from bytetrack.byte_tracker import BYTETracker

# =============================
# CONFIG & CLASSES (Giữ nguyên của bạn)
# =============================
ENGINE_PATH = "models/yolov8n_best_fp16.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.45
IOU_THRESH = 0.5
CLASSES = ["backpack", "handbag", "laptop", "person", "suitcase"]
COLORS = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 0, 255)]

# =============================
# CẤU HÌNH BYTETRACK
# =============================
class ByteTrackArgs:
    track_thresh = 0.45   # Threshold để bắt đầu track một vật
    track_buffer = 60     # Số frame ghi nhớ vật thể nếu nó bị che khuất
    match_thresh = 0.8    # Ngưỡng so khớp khoảng cách
    mot20 = False

# =============================
# TensorRT Class & GStreamer & Preprocess & Postprocess
# (Toàn bộ phần này GIỮ NGUYÊN NHƯ CODE CŨ CỦA BẠN - Mình ẩn đi cho gọn)
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

#def gstreamer_pipeline():
#    return ("nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=640,framerate=20/1 ! nvvidconv ! video/x-raw,width=640,height=640,format=BGRx ! appsink drop=1")

def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=640, framerate=30/1 ! "
        "nvvidconv ! video/x-raw, width=640, height=640, format=BGRx ! "
        "appsink drop=1 max-buffers=1"
    )
    
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
# Hàm Hỗ trợ ByteTrack: Tìm Class ban đầu
# =============================
def get_best_class(track_box, results):
    # ByteTrack mặc định không lưu Class ID. Hàm này đo khoảng cách 
    # từ box của tracker tới box của YOLO để lấy lại Class name.
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
# Hàm Vẽ Mới (CÓ TRACKING ID)
# =============================
def draw_tracked_boxes(frame, tracks, original_results, classes, colors):
    for track in tracks:
        if not track.is_activated: continue
        
        # ByteTrack trả về tọa độ [x1, y1, x2, y2]
        t_x1, t_y1, t_x2, t_y2 = track.tlbr
        x = int(t_x1)
        y = int(t_y1)
        w = int(t_x2 - t_x1)
        h = int(t_y2 - t_y1)
        track_id = track.track_id # LẤY ĐƯỢC ID TẠI ĐÂY
        
        # Tìm lại class_id từ kết quả gốc
        class_id = int(get_best_class([t_x1, t_y1, t_x2, t_y2], original_results))
        class_name = classes[class_id] if class_id < len(classes) else str(class_id)
        color = colors[class_id] if class_id < len(colors) else (255, 255, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # In tên class + ID
        label = f"ID:{track_id} {class_name}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - label_height - 5), (x + label_width, y), color, -1) 
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# =============================
# Main Loop Đã Cập Nhật
# =============================
def main():
    trt_infer = TensorRTInference(ENGINE_PATH)
    
    # 1. Khởi tạo Tracker ở ngoài vòng lặp
    tracker = BYTETracker(ByteTrackArgs())
    
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: continue

        if frame.shape[2] == 4:
            frame = frame[:, :, :3].copy()

        img = preprocess(frame)
        output = trt_infer.infer(img)
        
        # Kết quả YOLO gốc
        results = postprocess(output, frame, trt_infer.output_shape, CLASSES)

        # 2. Định dạng lại Input cho ByteTrack ([x1, y1, x2, y2, score])
        dets = []
        for x, y, w, h, conf, cls in results:
            dets.append([x, y, x+w, y+h, conf])
        dets = np.array(dets)

        # 3. Cập nhật Tracker
        if len(dets) > 0:
            img_info = [frame.shape[0], frame.shape[1]]
            # ByteTrack xử lý
            tracks = tracker.update(dets, img_info, img_info)
        else:
            tracks = []

        # 4. Vẽ kết quả Tracker
        draw_tracked_boxes(frame, tracks, results, CLASSES, COLORS)

        fps = 1/(time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)} | Tracker: ByteTrack", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Robot Tracking", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
