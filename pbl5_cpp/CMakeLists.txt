cmake_minimum_required(VERSION 3.10)
project(JetsonTracking)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    /usr/include/aarch64-linux-gnu
    /usr/include/eigen3
)

# THÊM DÒNG NÀY ĐỂ TRÌNH BIÊN DỊCH TÌM THẤY cudart
link_directories(
    /usr/local/cuda/lib64
)

file(GLOB SOURCES "src/*.cpp" "main.cpp")

add_executable(app ${SOURCES})

target_link_libraries(app
    ${OpenCV_LIBS}
    nvinfer
    nvparsers
    nvinfer_plugin
    cudart   # Giờ nó sẽ tìm thấy cudart nhờ thư mục link_directories ở trên
    pthread
)
