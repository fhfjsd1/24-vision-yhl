cmake_minimum_required(VERSION 3.10)
project(RectangleCenterDetection)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})#头文件搜索路径
link_directories(${OpenCV_LIBRARIES})#库文件搜索路径

add_executable(main main.cpp)

target_link_libraries(main PRIVATE ${OpenCV_LIBS})#链接动态库