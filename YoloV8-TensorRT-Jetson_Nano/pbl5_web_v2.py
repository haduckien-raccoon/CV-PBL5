import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from bytetrack.byte_tracker import BYTETracker

# =============================
# CONFIG
# =============================
ENGINE_PATH = "models/yolov8n_best_fp16.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.45
IOU_THRESH = 0.5

CLASSES = ["backpack", "handbag", "laptop", "person", "suitcase"]
COLORS = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (0, 0, 255)]

# =============================
# BYTE TRACK CONFIG
# =============================
class ByteTrackArgs:
    track_thresh = 0.45
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False

# =============================
# TENSORRT
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
# GSTREAMER PIPELINE (USB CAM)
# =============================
def gstreamer_pipeline():
    return (
        "v4l2src device=/dev/video1 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1 max-buffers=1"
    )

# =============================
# PREPROCESS
# =============================
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    return img

# =============================
# POSTPROCESS
# =============================
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
            results.append((detections[i][0], detections[i][1],
                            detections[i][2], detections[i][3],
                            scores_list[i], class_ids[i]))
    return results

# =============================
# BYTE TRACK HELPER
# =============================
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

# =============================
# DRAW
# =============================
def draw_tracked_boxes(frame, tracks, results):
    for track in tracks:
        if not track.is_activated:
            continue

        x1, y1, x2, y2 = track.tlbr
        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

        track_id = track.track_id

        class_id = int(get_best_class([x1, y1, x2, y2], results))
        class_name = CLASSES[class_id]

        color = COLORS[class_id]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        label = f"ID:{track_id} {class_name}"
        cv2.putText(frame, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# =============================
# MAIN
# =============================
def main():
    trt_infer = TensorRTInference(ENGINE_PATH)
    tracker = BYTETracker(ByteTrackArgs())

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    fps_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame error")
            continue

        img = preprocess(frame)
        output = trt_infer.infer(img)

        results = postprocess(output, frame, trt_infer.output_shape, CLASSES)

        dets = []
        for x, y, w, h, conf, cls in results:
            dets.append([x, y, x+w, y+h, conf])

        dets = np.array(dets)

        if len(dets) > 0:
            tracks = tracker.update(dets, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
        else:
            tracks = []

        draw_tracked_boxes(frame, tracks, results)

        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
