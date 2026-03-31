#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include "httplib.h"

std::mutex mtx;
std::vector<uchar> latest_jpeg;
std::atomic<bool> running(true);
std::atomic<double> measured_fps(0.0);  // atomic để đọc từ thread khác an toàn

// Hàm đo CPU sử dụng từ /proc/stat
double get_cpu_usage() {
    static long long last_idle = 0, last_total = 0;
    std::ifstream stat("/proc/stat");
    std::string line;
    std::getline(stat, line); // đọc dòng cpu tổng

    long long user, nice, system, idle, iowait, irq, softirq, steal;
    sscanf(line.c_str(), "cpu %lld %lld %lld %lld %lld %lld %lld %lld",
           &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal);

    long long idle_time = idle + iowait;
    long long total_time = user + nice + system + idle + iowait + irq + softirq + steal;

    long long diff_idle = idle_time - last_idle;
    long long diff_total = total_time - last_total;

    double cpu_usage = 0.0;
    if (diff_total > 0)
        cpu_usage = (1.0 - double(diff_idle) / diff_total) * 100.0;

    last_idle = idle_time;
    last_total = total_time;
    return cpu_usage;
}

void capture_loop(int cam, int w, int h, int fps) {
    cv::VideoCapture cap(cam, cv::CAP_V4L2);

    cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
    cap.set(cv::CAP_PROP_FPS, fps);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); // giảm lag

    if (!cap.isOpened()) {
        std::cerr << "Camera open failed\n";
        return;
    }

    auto last_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    while (running) {
        cv::Mat frame;
        if (!cap.read(frame)) continue;

        // Tính FPS
        frame_count++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        if (elapsed >= 1.0) {
            measured_fps.store(frame_count / elapsed, std::memory_order_relaxed);
            frame_count = 0;
            last_time = now;
        }

        // Vẽ FPS lên frame
        std::string fps_text = "FPS: " + std::to_string(int(measured_fps.load()));
        cv::putText(frame, fps_text, cv::Point(10,30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2);

        // Encode JPEG
        std::vector<uchar> buf;
        cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 80});

        // Lưu buffer thread-safe
        {
            std::lock_guard<std::mutex> lock(mtx);
            latest_jpeg = std::move(buf); // move để giảm copy
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// Thread in FPS + CPU ra console mỗi giây
void stats_loop() {
    while (running) {
        double fps = measured_fps.load();
        double cpu = get_cpu_usage();
        std::cout << "[Stats] FPS: " << int(fps)
                  << " | CPU: " << int(cpu) << "%\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main() {
    std::thread t(capture_loop, 0, 640, 480, 30);
    std::thread stats_thread(stats_loop);

    httplib::Server svr;

    svr.Get("/stream.mjpg", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "close");

        res.set_chunked_content_provider(
            "multipart/x-mixed-replace; boundary=frame",
            [](size_t, httplib::DataSink &sink) {
                while (running) {
                    std::vector<uchar> frame;

                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        frame = latest_jpeg;
                    }

                    if (!frame.empty()) {
                        std::string header =
                            "--frame\r\n"
                            "Content-Type: image/jpeg\r\n\r\n";

                        sink.write(header.c_str(), header.size());
                        sink.write((char*)frame.data(), frame.size());
                        sink.write("\r\n", 2);
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                return false;
            }
        );
    });

    std::cout << "Open: http://<IP>:8080/stream.mjpg\n";
    svr.listen("0.0.0.0", 8080);

    running = false;
    t.join();
    stats_thread.join();
}
