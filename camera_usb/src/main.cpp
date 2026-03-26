#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm> // Thêm thư viện này để dùng std::max

#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

// IMPORT BYTETRACK (Từ repo Qengineering / ByteTrack_with_labels)
#include "BYTETracker.h"

using namespace std;
using namespace nvinfer1;

// =============================
// CONFIG & CLASSES
// =============================
const string ENGINE_PATH = "models/yolov8n_best_fp16.engine";
const int INPUT_SIZE = 640;
const float CONF_THRESH = 0.45f;
const float IOU_THRESH = 0.5f;
const vector<string> CLASSES = {"backpack", "handbag", "laptop", "person", "suitcase"};
const vector<cv::Scalar> COLORS = {
    cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), 
    cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)
};

struct Detection {
    int x, y, w, h;
    float conf;
    int class_id;
};

// =============================
// 1. PIPELINE USB CAMERA (V4L2)
// =============================
// Nâng độ phân giải mặc định lên HD (1280x720) để lấy tối đa chi tiết
string usb_camera_pipeline(int device_id = 1, int width = 1280, int height = 720, int framerate = 30) {
    return "v4l2src device=/dev/video" + to_string(device_id) + " ! "
           "video/x-raw, width=(int)" + to_string(width) + ", height=(int)" + to_string(height) + ", framerate=(fraction)" + to_string(framerate) + "/1 ! "
           "videoconvert ! "
           "video/x-raw, format=(string)BGR ! "
           "appsink drop=1 max-buffers=1";
}

// =============================
// 2. ĐA LUỒNG CHO CAMERA 
// =============================
class VideoStream {
private:
    cv::VideoCapture cap;
    cv::Mat frame;
    std::mutex lock;
    std::atomic<bool> stopped{false};
    std::thread worker_thread;
    bool ret = false;

    void update() {
        while (!stopped) {
            cv::Mat temp_frame;
            bool success = cap.read(temp_frame);
            if (!success) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue; 
            }
            std::lock_guard<std::mutex> lk(lock);
            frame = temp_frame;
            ret = success;
        }
    }

public:
    VideoStream(const string& pipeline) {
        cap.open(pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            cerr << "[LỖI] Không thể mở Camera USB. Đang dùng cấu hình fallback..." << endl;
            cap.open(1, cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280); // Nâng HD
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720); // Nâng HD
            cap.set(cv::CAP_PROP_FPS, 30);
        }
        if (cap.isOpened()) {
            ret = cap.read(frame);
        }
    }

    ~VideoStream() {
        stopped = true;
        if (worker_thread.joinable()) worker_thread.join();
        if (cap.isOpened()) cap.release();
    }

    VideoStream* start() {
        worker_thread = std::thread(&VideoStream::update, this);
        return this;
    }

    bool read(cv::Mat& frame_out) {
        std::lock_guard<std::mutex> lk(lock);
        if (!ret || frame.empty()) return false;
        frame_out = frame.clone();
        return true;
    }

    void stop() { stopped = true; }
};

// =============================
// 3. TENSORRT INFERENCE 
// =============================
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) cout << msg << endl; 
    }
} gLogger;

class TensorRTInference {
private:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;
    void* buffers[2]; 
    int input_idx = 0;
    int output_idx = 1;
    size_t input_size;
    size_t output_size;

public:
    vector<float> host_output;
    int out_dim_2 = 1;

    TensorRTInference(const string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) throw std::runtime_error("Không tìm thấy model engine!");

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);
        file.close();

        runtime = createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
        context = engine->createExecutionContext();

        Dims in_dims = engine->getBindingDimensions(input_idx);
        Dims out_dims = engine->getBindingDimensions(output_idx);

        int in_volume = 1; for (int i = 0; i < in_dims.nbDims; i++) in_volume *= in_dims.d[i];
        int out_volume = 1; for (int i = 0; i < out_dims.nbDims; i++) out_volume *= out_dims.d[i];
        if (out_dims.nbDims == 3) out_dim_2 = out_dims.d[2]; 

        input_size = in_volume * sizeof(float);
        output_size = out_volume * sizeof(float);

        cudaMalloc(&buffers[input_idx], input_size);
        cudaMalloc(&buffers[output_idx], output_size);
        cudaStreamCreate(&stream);

        host_output.resize(out_volume);
    }
    
    ~TensorRTInference() {
        cudaStreamDestroy(stream);
        cudaFree(buffers[input_idx]);
        cudaFree(buffers[output_idx]);
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }

    void infer(const vector<float>& input_data) {
        cudaMemcpyAsync(buffers[input_idx], input_data.data(), input_size, cudaMemcpyHostToDevice, stream);
        context->enqueueV2(buffers, stream, nullptr);
        cudaMemcpyAsync(host_output.data(), buffers[output_idx], output_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
};

// =============================
// 4. PRE/POST PROCESS (ĐÃ FIX LETTERBOX)
// =============================

// Hàm tạo viền xám (Letterbox) để giữ nguyên tỷ lệ ảnh
cv::Mat format_to_square(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row); // Tìm cạnh dài nhất
    // Tạo ảnh vuông nền xám
    cv::Mat result(cv::Size(_max, _max), CV_8UC3, cv::Scalar(114, 114, 114));
    // Dán ảnh gốc vào góc trên trái
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

vector<float> preprocess(const cv::Mat& frame) {
    // 1. Chèn viền xám để thành ảnh vuông
    cv::Mat square_img = format_to_square(frame);
    cv::Mat blob;
    // 2. Thu nhỏ ảnh vuông về 640x640 cho model (lúc này tỷ lệ vật thể được giữ nguyên)
    cv::dnn::blobFromImage(square_img, blob, 1.0 / 255.0, cv::Size(INPUT_SIZE, INPUT_SIZE), cv::Scalar(), true, false);
    vector<float> input_data((float*)blob.datastart, (float*)blob.dataend);
    return input_data;
}

vector<Detection> postprocess(const vector<float>& output, int img_w, int img_h, int num_anchors) {
    int num_classes = CLASSES.size();
    vector<cv::Rect> boxes;
    vector<float> scores;
    vector<int> class_ids;

    // Tính tỷ lệ scale ngược từ 640x640 về ảnh gốc
    float scale_factor = std::max(img_w, img_h) / (float)INPUT_SIZE;

    for (int i = 0; i < num_anchors; i++) {
        float max_score = 0;
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
            float bw  = output[2 * num_anchors + i];
            float bh  = output[3 * num_anchors + i];

            // Nhân ngược tọa độ với scale_factor để lấy tọa độ chuẩn trên khung hình 1280x720
            int x1 = std::max(0, (int)((cx - bw / 2.0f) * scale_factor));
            int y1 = std::max(0, (int)((cy - bh / 2.0f) * scale_factor));
            int width = (int)(bw * scale_factor);
            int height = (int)(bh * scale_factor);

            boxes.push_back(cv::Rect(x1, y1, width, height));
            scores.push_back(max_score);
            class_ids.push_back(best_class);
        }
    }

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH, indices);

    vector<Detection> results;
    for (int idx : indices) {
        Detection det;
        det.x = boxes[idx].x; det.y = boxes[idx].y;
        det.w = boxes[idx].width; det.h = boxes[idx].height;
        det.conf = scores[idx];
        det.class_id = class_ids[idx];
        results.push_back(det);
    }
    return results;
}

int get_best_class(const vector<float>& t_box, const vector<Detection>& results) {
    float tx_center = (t_box[0] + t_box[2]) / 2.0f;
    float ty_center = (t_box[1] + t_box[3]) / 2.0f;
    int best_class = 0;
    float min_dist = 1e9f;

    for (const auto& res : results) {
        float rx_center = res.x + res.w / 2.0f;
        float ry_center = res.y + res.h / 2.0f;
        float dist = std::pow(tx_center - rx_center, 2) + std::pow(ty_center - ry_center, 2);
        if (dist < min_dist) {
            min_dist = dist;
            best_class = res.class_id;
        }
    }
    return best_class;
}

void draw_tracked_boxes(cv::Mat& frame, const vector<STrack>& tracks, const vector<Detection>& original_results) {
    for (const auto& track : tracks) {
        if (!track.is_activated) continue;
        
        const float* tlwh = (const float*)&track.tlwh;
        int x = (int)tlwh[0];
        int y = (int)tlwh[1];
        int w = (int)tlwh[2];
        int h = (int)tlwh[3];
        int track_id = track.track_id;

        float t_x1 = tlwh[0];
        float t_y1 = tlwh[1];
        float t_x2 = tlwh[0] + tlwh[2];
        float t_y2 = tlwh[1] + tlwh[3];
        
        vector<float> box_coords = {t_x1, t_y1, t_x2, t_y2};
        int class_id = get_best_class(box_coords, original_results);
        
        string class_name = (class_id < CLASSES.size()) ? CLASSES[class_id] : to_string(class_id);
        cv::Scalar color = (class_id < COLORS.size()) ? COLORS[class_id] : cv::Scalar(255, 255, 255);

        cv::rectangle(frame, cv::Rect(x, y, w, h), color, 2);
        string label = "ID:" + to_string(track_id) + " " + class_name;
        
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(frame, cv::Rect(x, y - labelSize.height - 5, labelSize.width, labelSize.height + 5), color, cv::FILLED);
        cv::putText(frame, label, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

// =============================
// 5. MAIN LOOP 
// =============================
int main() {
    cout << "[INFO] Khởi động TensorRT Engine..." << endl;
    TensorRTInference trt_infer(ENGINE_PATH);
    
    BYTETracker tracker(30, 60);

    cout << "[INFO] Khởi động luồng Camera USB ở chế độ HD (1280x720)..." << endl;
    VideoStream* video_stream = new VideoStream(usb_camera_pipeline(1, 1280, 720, 30));
    video_stream->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    auto fps_time = std::chrono::high_resolution_clock::now();
    int frames_count = 0;
    float current_fps = 0.0f;

    cout << "[INFO] Đang chạy Inference..." << endl;
    while (true) {
        cv::Mat frame;
        bool ret = video_stream->read(frame);
        if (!ret || frame.empty()) continue;

        vector<float> img = preprocess(frame);
        trt_infer.infer(img);
        
        vector<Detection> results = postprocess(trt_infer.host_output, frame.cols, frame.rows, trt_infer.out_dim_2);

        vector<Object> objects;
        for (const auto& det : results) {
            Object obj;
            obj.rect = cv::Rect_<float>(det.x, det.y, det.w, det.h);
            obj.prob = det.conf;
            obj.label = det.class_id;
            objects.push_back(obj);
        }

        vector<STrack> tracks;
        if (!objects.empty()) {
            tracks = tracker.update(objects);
        }

        draw_tracked_boxes(frame, tracks, results);

        frames_count++;
        if (frames_count % 10 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> elapsed = now - fps_time;
            current_fps = 10.0f / elapsed.count();
            fps_time = now;
            frames_count = 0;
        }

        cv::putText(frame, "FPS: " + to_string((int)current_fps) + " | Tracker: ByteTrack", 
                    cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("USB Camera Tracking", frame);
        if (cv::waitKey(1) == 27) break; 
    }

    cout << "[INFO] Đang tắt hệ thống..." << endl;
    video_stream->stop();
    delete video_stream;
    cv::destroyAllWindows();

    return 0;
}
