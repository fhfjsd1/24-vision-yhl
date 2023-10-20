cmake_minimum_required(VERSION 3.13)

project(ONNX_Runtime_main VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)
include_directories(/home/taylor/testcpp/cpp-onnxruntime-resnet-console-app-main/OnnxRuntimeResNet/)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/include/onnxruntime-linux-x64-gpu-1.16.1/include/)

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/include/onnxruntime-linux-x64-gpu-1.16.1/lib)


add_executable(main OnnxRuntimeResNet/OnnxRuntimeResNet.cpp OnnxRuntimeResNet/Helpers.cpp)
target_include_directories(main PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})



