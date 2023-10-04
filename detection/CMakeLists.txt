cmake_minimum_required(VERSION 3.10)
project(RectangleCenterDetection)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES})
add_executable(main main.cpp)

target_link_libraries(main PRIVATE ${OpenCV_LIBS})
