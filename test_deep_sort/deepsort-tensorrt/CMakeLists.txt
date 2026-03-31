cmake_minimum_required(VERSION 3.10) # Nâng lên 3.10 để hỗ trợ TensorRT tốt hơn
project(deepsort_yolo_pbl5)

# Sử dụng chuẩn C++14 để hỗ trợ tốt nhất cho TensorRT và các hàm hiện đại
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# Tìm các thư viện cần thiết
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Thêm các thư mục chứa header
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    /usr/include/eigen3
)

# Đường dẫn thư viện CUDA
link_directories(
    /usr/local/cuda/lib64
)

# Tự động lấy tất cả các file .cpp trong thư mục src
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_DIR)

# 1. Tạo thư viện dùng chung deepsort (Kalman Filter, Matching, v.v.)
add_library(deepsort SHARED ${SRC_DIR})
target_link_libraries(deepsort 
    ${CUDA_LIBS} 
    ${OpenCV_LIBS} 
    cudart nvinfer nvonnxparser
)

# 2. Tạo file thực thi chính của bạn (pbl5_app)
# File này sẽ kết hợp YOLOv8 và thư viện deepsort ở trên
add_executable(pbl5_app main.cpp)
target_link_libraries(pbl5_app
    ${CUDA_LIBS}
    ${OpenCV_LIBS}
    deepsort  # Liên kết với thư viện deepsort vừa tạo ở trên
    cudart 
    nvinfer 
    nvonnxparser
    pthread
)

# (Tùy chọn) Giữ lại các công cụ demo và convert model
add_executable(onnx2engine ${PROJECT_SOURCE_DIR}/onnx2engine.cpp)
target_link_libraries(onnx2engine nvinfer nvonnxparser deepsort)

add_executable(demo ${PROJECT_SOURCE_DIR}/demo.cpp)
target_link_libraries(demo ${OpenCV_LIBS} deepsort)
