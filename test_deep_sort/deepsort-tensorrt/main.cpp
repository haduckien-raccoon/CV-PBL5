#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

// Include header của DeepSORT thay vì ByteTrack
#include "deepsort.h"

using namespace std;
using namespace nvinfer1;

// =============================
// CONFIG
// =============================
const string YOLO_ENGINE_PATH = "models/yolov8n_best_fp16.engine";
const string DEEPSORT_ENGINE_PATH = "resources/deepsort.engine"; 

const int INPUT_SIZE = 640;
const float CONF_THRESH = 0.45f;
const float IOU_THRESH = 0.5f;

const vector<string> CLASSES = {"backpack", "handbag", "laptop", "person", "suitcase"};
const vector<cv::Scalar> COLORS = {
    cv::Scalar(255, 0, 0),    // backpack
    cv::Scalar(0, 255, 255),  // handbag
    cv::Scalar(255, 0, 255),  // laptop
    cv::Scalar(0, 255, 0),    // person
    cv::Scalar(0, 0, 255)     // suitcase
};

struct Detection {
    int x, y, w, h;
    float conf;
    int class_id;
};

// =============================
// 1) CAMERA STREAM (THREAD)
// =============================
class VideoStream {
private:
    cv::VideoCapture cap;
    cv::Mat frame;
    mutable std::mutex mtx;
    std::atomic<bool> stopped{false};
    std::thread worker_thread;
    bool ret = false;

    void update() {
        while (!stopped) {
            cv::Mat temp;
            bool success = cap.read(temp);
            if (!success) {
                stopped = true;
                break;
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                frame = temp;
                ret = true;
            }
        }
    }

public:
    explicit VideoStream(const string& pipeline) {
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            throw runtime_error("[ERROR] Khong mo duoc camera voi GStreamer pipeline.");
        }
        ret = cap.read(frame);
        if (!ret || frame.empty()) {
            throw runtime_error("[ERROR] Khong doc duoc frame dau tien tu camera.");
        }
    }

    ~VideoStream() {
        stop();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        if (cap.isOpened()) {
            cap.release();
        }
    }

    void start() {
        worker_thread = std::thread(&VideoStream::update, this);
    }

    bool read(cv::Mat& out) {
        std::lock_guard<std::mutex> lk(mtx);
        if (!ret || frame.empty()) return false;
        out = frame.clone();
        return true;
    }

    void stop() {
        stopped = true;
    }
};

// =============================
// 2) TENSORRT YOLO
// =============================
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        (void)severity;
        (void)msg;
    }
} gLogger;

class TensorRTInference {
private:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream{};

    void* buffers[2] = {nullptr, nullptr};
    int input_idx = 0;
    int output_idx = 1;
    size_t input_size = 0;
    size_t output_size = 0;

public:
    vector<float> host_output;
    int out_dim_2 = 1; // so anchors

    explicit TensorRTInference(const string& engine_path) {
        ifstream file(engine_path, ios::binary);
        if (!file.good()) throw runtime_error("[ERROR] Khong tim thay TensorRT engine: " + engine_path);

        file.seekg(0, file.end);
        size_t size = static_cast<size_t>(file.tellg());
        file.seekg(0, file.beg);

        vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);
        file.close();

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
        context = engine->createExecutionContext();

        // [FIX 1] Tự động tìm Binding Index của Input và Output cho TensorRT 8
        for (int i = 0; i < engine->getNbBindings(); i++) {
            if (engine->bindingIsInput(i)) {
                input_idx = i;
            } else {
                output_idx = i;
            }
        }

        Dims in_dims = engine->getBindingDimensions(input_idx);
        Dims out_dims = engine->getBindingDimensions(output_idx);

        int in_volume = 1;
        for (int i = 0; i < in_dims.nbDims; i++) in_volume *= in_dims.d[i];
        int out_volume = 1;
        for (int i = 0; i < out_dims.nbDims; i++) out_volume *= out_dims.d[i];

        if (out_dims.nbDims == 3) out_dim_2 = out_dims.d[2];

        input_size = static_cast<size_t>(in_volume) * sizeof(float);
        output_size = static_cast<size_t>(out_volume) * sizeof(float);

        cudaMalloc(&buffers[input_idx], input_size);
        cudaMalloc(&buffers[output_idx], output_size);
        cudaStreamCreate(&stream);

        host_output.resize(out_volume);
    }

    ~TensorRTInference() {
        if (stream) cudaStreamDestroy(stream);
        if (buffers[input_idx]) cudaFree(buffers[input_idx]);
        if (buffers[output_idx]) cudaFree(buffers[output_idx]);

        if (context) { delete context; context = nullptr; }
        if (engine)  { delete engine;  engine = nullptr; }
        if (runtime) { delete runtime; runtime = nullptr; }
    }

    void infer(const vector<float>& input_data) {
        cudaMemcpyAsync(buffers[input_idx], input_data.data(), input_size, cudaMemcpyHostToDevice, stream);
        context->enqueueV2(buffers, stream, nullptr);
        cudaMemcpyAsync(host_output.data(), buffers[output_idx], output_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
};

// =============================
// 3) PIPELINE + PRE/POST
// =============================
string gstreamer_pipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate) {
    return "nvarguscamerasrc ! "
           "video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) +
           ", height=(int)" + to_string(capture_height) +
           ", format=(string)NV12, framerate=(fraction)" + to_string(framerate) + "/1 ! "
           "nvvidconv interpolation-method=1 ! "
           "video/x-raw, width=(int)" + to_string(display_width) +
           ", height=(int)" + to_string(display_height) + ", format=(string)BGRx ! "
           "videoconvert ! "
           "video/x-raw, format=(string)BGR ! "
           "appsink drop=1 max-buffers=1";
}

vector<float> preprocess(const cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(), true, false);
    return vector<float>((float*)blob.datastart, (float*)blob.dataend);
}

vector<Detection> postprocess(const vector<float>& output, int img_w, int img_h, int num_anchors) {
    const int num_classes = static_cast<int>(CLASSES.size());
    vector<cv::Rect> boxes;
    vector<float> scores;
    vector<int> class_ids;

    boxes.reserve(num_anchors);
    scores.reserve(num_anchors);
    class_ids.reserve(num_anchors);

    for (int i = 0; i < num_anchors; i++) {
        float max_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes; c++) {
            float score = output[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        if (max_score > CONF_THRESH) {
            float cx = output[0 * num_anchors + i];
            float cy = output[1 * num_anchors + i];
            float bw = output[2 * num_anchors + i];
            float bh = output[3 * num_anchors + i];

            int x1 = max(0, static_cast<int>((cx - bw / 2.0f) * img_w / INPUT_SIZE));
            int y1 = max(0, static_cast<int>((cy - bh / 2.0f) * img_h / INPUT_SIZE));
            int w = static_cast<int>(bw * img_w / INPUT_SIZE);
            int h = static_cast<int>(bh * img_h / INPUT_SIZE);

            boxes.emplace_back(x1, y1, w, h);
            scores.push_back(max_score);
            class_ids.push_back(best_class);
        }
    }

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH, indices);

    vector<Detection> results;
    results.reserve(indices.size());
    for (int idx : indices) {
        results.push_back({boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height, scores[idx], class_ids[idx]});
    }

    return results;
}

// Hàm vẽ cho DeepSORT
void draw_deepsort_boxes(cv::Mat& frame, const vector<DetectBox>& tracked_boxes) {
    for (const auto& box : tracked_boxes) {
        int track_id = box.trackID; 
        int class_id = box.classID;

        if (track_id <= 0) continue; 

        int x = box.x1;
        int y = box.y1;
        int w = box.x2 - box.x1;
        int h = box.y2 - box.y1;

        string class_name = (class_id >= 0 && class_id < (int)CLASSES.size()) ? CLASSES[class_id] : to_string(class_id);
        cv::Scalar color = (class_id >= 0 && class_id < (int)COLORS.size()) ? COLORS[class_id] : cv::Scalar(255, 255, 255);

        cv::rectangle(frame, cv::Rect(x, y, w, h), color, 2);

        string label = "ID:" + to_string(track_id) + " " + class_name;
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int box_y = max(0, y - labelSize.height - 6);
        cv::rectangle(frame, cv::Rect(x, box_y, labelSize.width + 4, labelSize.height + 6), color, cv::FILLED);
        cv::putText(frame, label, cv::Point(x + 2, box_y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// =============================
// 4) UTIL: parse max fps
// =============================
float parse_max_fps(int argc, char** argv, float default_fps = 10.0f) {
    float max_fps = default_fps;
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--max-fps" && i + 1 < argc) {
            max_fps = std::stof(argv[++i]);
        } else if (arg.rfind("--max-fps=", 0) == 0) {
            max_fps = std::stof(arg.substr(string("--max-fps=").size()));
        }
    }
    if (max_fps <= 0.0f) {
        cerr << "[WARN] max_fps <= 0, fallback ve 10." << endl;
        max_fps = 10.0f;
    }
    return max_fps;
}

// =============================
// 5) MAIN
// =============================
int main(int argc, char** argv) {
    try {
        float max_fps = parse_max_fps(argc, argv, 10.0f);
        const auto target_frame_time = std::chrono::duration<double>(1.0 / max_fps);

        cout << "[INFO] max_fps = " << max_fps << endl;
        
        cout << "[INFO] Khoi dong YOLO TensorRT Engine..." << endl;
        TensorRTInference trt_infer(YOLO_ENGINE_PATH);

        cout << "[INFO] Khoi dong DeepSORT Tracker..." << endl;
        DeepSort tracker(DEEPSORT_ENGINE_PATH, 128, 512, 0, &gLogger);
        
        int camera_fps = static_cast<int>(std::round(max_fps));
        camera_fps = std::max(1, camera_fps);

        cout << "[INFO] Khoi dong camera thread..." << endl;
        VideoStream video_stream(gstreamer_pipeline(640, 640, 640, 640, camera_fps));
        video_stream.start();

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto fps_window_start = std::chrono::steady_clock::now();
        int fps_window_frames = 0;
        float display_fps = 0.0f;

        cout << "[INFO] Dang chay inference..." << endl;
        while (true) {
            auto loop_start = std::chrono::steady_clock::now();

            cv::Mat frame;
            if (!video_stream.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            cv::Mat frame_bgr;
            if (frame.channels() == 4) {
                cv::cvtColor(frame, frame_bgr, cv::COLOR_BGRA2BGR);
            } else {
                frame_bgr = frame;
            }

            // 1. YOLOv8 Inference
            vector<float> input = preprocess(frame_bgr);
            trt_infer.infer(input);
            vector<Detection> results = postprocess(
                trt_infer.host_output,
                frame_bgr.cols,
                frame_bgr.rows,
                trt_infer.out_dim_2
            );

            // 2. Chuyển đổi định dạng và khóa tọa độ cho DeepSORT
            vector<DetectBox> ds_boxes;
            ds_boxes.reserve(results.size());
            for (const auto& det : results) {
                DetectBox box;
                box.classID = det.class_id;
                box.confidence = det.conf;
                
                // [FIX 2] Khóa chặt tọa độ vào đúng kích thước khung hình
                box.x1 = std::max(0, det.x);
                box.y1 = std::max(0, det.y);
                box.x2 = std::min(frame_bgr.cols - 1, det.x + det.w);
                box.y2 = std::min(frame_bgr.rows - 1, det.y + det.h);
                
                // Loại bỏ những box không hợp lệ (ví dụ chiều rộng <= 0)
                if (box.x1 >= box.x2 || box.y1 >= box.y2) continue;

                ds_boxes.push_back(box);
            }

            // 3. Update Tracker (DeepSORT tự động crop ảnh, trích xuất Feature và gán ID)
            if (!ds_boxes.empty()) {
                tracker.sort(frame_bgr, ds_boxes);
            }

            // 4. Vẽ kết quả
            draw_deepsort_boxes(frame_bgr, ds_boxes);

            // Tính FPS
            fps_window_frames++;
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<float> elapsed = now - fps_window_start;
            if (elapsed.count() >= 1.0f) {
                display_fps = fps_window_frames / elapsed.count();
                fps_window_frames = 0;
                fps_window_start = now;
            }

            cv::putText(frame_bgr,
                        "FPS: " + to_string((int)std::round(display_fps)) + " | Max: " + to_string((int)std::round(max_fps)) + " | DeepSORT",
                        cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Robot Tracking Multi-Thread", frame_bgr);
            int key = cv::waitKey(1);
            if (key == 27) break; // ESC

            // HARD CAP FPS
            auto loop_end = std::chrono::steady_clock::now();
            auto loop_elapsed = loop_end - loop_start;
            if (loop_elapsed < target_frame_time) {
                auto sleep_time = std::chrono::duration_cast<std::chrono::milliseconds>(target_frame_time - loop_elapsed);
                if (sleep_time.count() > 0) {
                    std::this_thread::sleep_for(sleep_time);
                }
            }
        }

        cout << "[INFO] Tat he thong..." << endl;
        video_stream.stop();
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& e) {
        cerr << e.what() << endl;
        return -1;
    }
}
