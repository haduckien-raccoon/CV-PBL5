#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Mở camera mặc định
    VideoCapture cap(1);
    if(!cap.isOpened()) {
        cerr << "ERROR: Không mở được webcam!" << endl;
        return -1;
    }

    // Có thể đặt độ phân giải (tuỳ ý)
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame;
    while(true) {
        cap >> frame; // đọc frame từ camera
        if(frame.empty()) {
            cerr << "ERROR: Không đọc được frame!" << endl;
            break;
        }

        imshow("Webcam Test", frame); // hiển thị frame

        char key = (char)waitKey(30);
        if(key == 27) break; // nhấn ESC để thoát
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
