# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.10)
project(modules)

#查找该目录下的所有源文件

file(GLOB SRC1 ${CMAKE_CURRENT_SOURCE_DIR}/A1/src/*.cpp)
file(GLOB SRC2 ${CMAKE_CURRENT_SOURCE_DIR}/A2/src/*.cpp)
file(GLOB SRC3 ${CMAKE_CURRENT_SOURCE_DIR}/M1/src/*.cpp)
file(GLOB SRC4 ${CMAKE_CURRENT_SOURCE_DIR}/M2/src/*.cpp)

set(SRC ${SRC1} ${SRC2} ${SRC3} ${SRC4})

#生成静态库
add_library(modules STATIC ${SRC})

#链接这个库的其他依赖项
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES})
target_link_libraries(modules  ${OpenCV_LIBRARIES})