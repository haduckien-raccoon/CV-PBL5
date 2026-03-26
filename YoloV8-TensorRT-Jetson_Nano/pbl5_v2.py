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

# Danh sách class chuẩn của bạn (BẮT BUỘC ĐÚNG THỨ TỰ lúc train)
CLASSES = ["backpack", "handbag", "laptop", "person", "suitcase"]

# Bảng màu tương ứng cho 5 class (Mã màu BGR của OpenCV)
# Mỗi class sẽ có một màu sắc riêng để dễ phân biệt
COLORS = [
    (255, 0, 0),    # backpack: Xanh dương
    (0, 255, 255),  # handbag: Vàng
    (255, 0, 255),  # laptop: Hồng tím
    (0, 255, 0),    # person: Xanh lá
    (0, 0, 255)     # suitcase: Đỏ
]

# =============================
# TensorRT Class (Quản lý Engine)
# =============================
class TensorRTInference:
    def __init__(self, engine_path):
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Deserialize engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Lấy binding index và shape
        self.input_binding_idx = 0
        self.output_binding_idx = 1
        
        self.input_shape = tuple(self.engine.get_binding_shape(self.input_binding_idx))
        self.output_shape = tuple(self.engine.get_binding_shape(self.output_binding_idx))
        
        # Tính toán kích thước bộ nhớ (volume)
        # abs() dùng để xử lý explicit batch dimension (1,3,640,640) -> 1*3*640*640
        self.input_volume = abs(trt.volume(self.input_shape))
        self.output_volume = abs(trt.volume(self.output_shape))
        
        # Cấp phát bộ nhớ GPU (4 bytes cho FP32/FP16 dữ liệu đầu ra)
        self.d_input = cuda.mem_alloc(self.input_volume * 4)
        self.d_output = cuda.mem_alloc(self.output_volume * 4)
        
        self.bindings = [int(self.d_input), int(self.d_output)]
        
        # Tạo stream và host output buffer
        self.stream = cuda.Stream()
        self.host_output = np.empty(self.output_volume, dtype=np.float32)
        
        print(f"Engine input shape: {self.input_shape}")
        print(f"Engine output shape: {self.output_shape}")

    def infer(self, input_data):
        # 1. Copy dữ liệu preprocess từ Host sang Device (GPU)
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # 2. Chạy inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # 3. Copy kết quả từ Device về Host
        cuda.memcpy_dtoh_async(self.host_output, self.d_output, self.stream)
        
        # 4. Đồng bộ hóa stream
        self.stream.synchronize()
        
        return self.host_output

# =============================
# GStreamer Pipeline tối ưu
# =============================
def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM),width=640,height=640,framerate=50/1 ! "
        "nvvidconv ! "
        "video/x-raw,width=640,height=640,format=BGRx ! "
        "appsink drop=1"
    )

# =============================
# Logic Xử lý (Postprocess & Vẽ)
# =============================
def preprocess(frame):
    # Resize về 640x640
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    # BGR to RGB
    img = img[:, :, ::-1] 
    # HWC to CHW
    img = img.transpose(2, 0, 1) 
    # Chuyển kiểu dữ liệu CONTIGUOUS và bình thường hóa về 0-1
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    return img

def postprocess(output, frame, out_shape, classes):
    h, w = frame.shape[:2]
    num_classes = len(classes)
    
    # Reshape dựa trên shape thật của engine và bỏ dimension batch (1, 9, 8400) -> (9, 8400)
    output = output.reshape(out_shape).squeeze()

    # FIX TỌA ĐỘ LOẠN: YOLOv8 trả về (num_anchors, coordinates+classes) - e.g., (8400, 9)
    # Ta bắt buộc phải transpose về (coordinates+classes, num_anchors) - e.g., (9, 8400)
    if output.shape[0] == 8400:
        output = output.T

    # 4 Tọa độ đầu tiên: cx, cy, w, h (px)
    boxes = output[0:4, :]
    # Các dòng tiếp theo là score của các class
    scores = output[4:(4+num_classes), :]

    # Lấy class ID và confidence cao nhất cho mỗi anchor
    class_ids = np.argmax(scores, axis=0)
    confidences = np.max(scores, axis=0)

    # Lọc theo CONF_THRESH
    mask = confidences > CONF_THRESH

    boxes = boxes[:, mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    detections = []
    scores_list = []

    # Scale tọa độ box về kích thước khung hình gốc
    # Tọa độ đang ở dạng cx, cy, bw, bh relative to input size (640)
    for i in range(boxes.shape[1]):
        x, y, bw, bh = boxes[:,i]
        
        # Scale & Chuyển sang format (x1, y1, width, height)
        x1 = int((x - bw/2) * w / INPUT_SIZE)
        y1 = int((y - bh/2) * h / INPUT_SIZE)
        
        # Giới hạn coordinates trong phạm vi ảnh
        x1 = max(0, x1)
        y1 = max(0, y1)
        width = int(bw * w / INPUT_SIZE)
        height = int(bh * h / INPUT_SIZE)

        detections.append([x1, y1, width, height])
        scores_list.append(float(confidences[i]))

    # Chạy NMS (Non-Maximum Suppression) để loại bỏ box đè nhau
    idxs = cv2.dnn.NMSBoxes(detections, scores_list, CONF_THRESH, IOU_THRESH)
    
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            results.append((detections[i][0], detections[i][1], detections[i][2], detections[i][3], scores_list[i], class_ids[i]))
            
    return results

def draw_boxes(frame, results, classes, colors):
    for x, y, w, h, conf, cls in results:
        class_id = int(cls)
        
        # Lấy tên và màu sắc tương ứng
        if class_id < len(classes):
            class_name = classes[class_id]
            color = colors[class_id]
        else:
            class_name = str(class_id)
            color = (255, 255, 255) # Trắng mặc định

        # Vẽ khung chữ nhật với màu riêng
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Vẽ nền mờ cho chữ dễ đọc
        label = f"{class_name} {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Vẽ nền với màu của class, opacity giả bằng cách vẽ đè
        cv2.rectangle(frame, (x, y - label_height - 5), (x + label_width, y), color, -1) 
        # Vẽ chữ màu đen
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# =============================
# Main Loop
# =============================
def main():
    print("Vui lòng đảm bảo đã chạy: 'export DISPLAY=:1' (hoặc :0 tùy VNC config)")
    print("Và 'sudo jetson_clocks' để Max-N hiệu năng.")
    
    # 1. Khởi tạo TensorRT
    trt_infer = TensorRTInference(ENGINE_PATH)
    
    # 2. Khởi tạo Camera
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Camera failed")
        return

    # 3. Đo FPS
    fps_time = time.time()

    # 4. Vòng lặp nhận diện
    while True:
        ret, frame = cap.read()
        if not ret: continue

        # Chuyển format đầu ra GStreamer BGRx (4 kênh) sang BGR (3 kênh) tốc độ cao
        if frame.shape[2] == 4:
            frame = frame[:, :, :3].copy()

        # Tiền xử lý (Host)
        img = preprocess(frame)

        # Suy luận (GPU)
        output = trt_infer.infer(img)

        # Hậu xử lý (Host) - Giải quyết dứt điểm lỗi loạn box
        results = postprocess(output, frame, trt_infer.output_shape, CLASSES)

        # Vẽ kết quả (Host)
        draw_boxes(frame, results, CLASSES, COLORS)

        # Tính toán và hiển thị FPS
        fps = 1/(time.time() - fps_time)
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # HIỂN THỊ LÊN MÀN HÌNH VNC
        cv2.imshow("YOLOv8 TensorRT (Cắm màn hình trực tiếp để mượt nhất)", frame)

        # Nhấn ESC để thoát
        if cv2.waitKey(1) == 27:
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
