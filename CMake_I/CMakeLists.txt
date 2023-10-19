# 在此处进行 CMake 设置（最小版本号，项目名）
cmake_minimum_required(VERSION 3.10)
project(Test)

#指定可能用到的头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules/A1/include 
${CMAKE_CURRENT_SOURCE_DIR}/modules/A2/include 
${CMAKE_CURRENT_SOURCE_DIR}/modules/M1/include
${CMAKE_CURRENT_SOURCE_DIR}/modules/M2/include
${CMAKE_CURRENT_SOURCE_DIR}/common/kalman/include
${CMAKE_CURRENT_SOURCE_DIR}/common/math/include
${OpenCV_INCLUDE_DIRS})

####### 以下禁止修改 #######
add_subdirectory(common)   #没有修改
add_subdirectory(modules)  #
####### 以上禁止修改 #######

# 在此处链接静态库并生成可执行程序,若生成动态库则放在后面再链接
link_libraries(modules common)
add_executable(main main.cpp )

# 在此处链接其他库
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARIES})
target_link_libraries(main  ${OpenCV_LIBRARIES})
