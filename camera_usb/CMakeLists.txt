cmake_minimum_required(VERSION 3.10)
project(pbl5_cpp)

# Khai báo chuẩn C++ (ByteTrack thường yêu cầu C++14 trở lên)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. TÌM VÀ LIÊN KẾT THƯ VIỆN BÊN NGOÀI
# ==========================================

# Tìm OpenCV (Bản cài trên Jetson Nano đã tích hợp sẵn GStreamer)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Tìm CUDA Toolkit
find_package(CUDA REQUIRED)
message(STATUS "CUDA version: ${CUDA_VERSION}")

# ==========================================
# 2. KHAI BÁO THƯ MỤC CHỨA FILE HEADER (.h)
# ==========================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include   # Thư mục include của dự án chứa file ByteTrack
    ${OpenCV_INCLUDE_DIRS}                # Header của OpenCV
    ${CUDA_INCLUDE_DIRS}                  # Header của CUDA
    /usr/include/aarch64-linux-gnu        # Thư mục mặc định chứa header TensorRT trên Jetson Nano
    /usr/include/eigen3                   # <--- THÊM DÒNG NÀY ĐỂ FIX LỖI EIGEN
)

# ==========================================
# 3. GOM MÃ NGUỒN VÀ BIÊN DỊCH
# ==========================================
# Tự động lấy tất cả các file .cpp nằm trong thư mục src/
file(GLOB SOURCES "src/*.cpp")

# Chỉ đường dẫn thẳng tới thư viện CUDA cho bộ liên kết (Linker)
link_directories(/usr/local/cuda/lib64)
# Tạo file thực thi tên là 'app' từ các file mã nguồn trên
add_executable(app ${SOURCES})

# ==========================================
# 4. LIÊN KẾT FILE CHẠY VỚI THƯ VIỆN CỐT LÕI
# ==========================================
target_link_libraries(app
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer           # Thư viện cốt lõi của TensorRT
    nvinfer_plugin    # Các plugin hỗ trợ của TensorRT (nếu YOLOv8 có dùng custom layer)
    cudart            # CUDA Runtime API
)
