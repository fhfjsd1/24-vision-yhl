cmake_minimum_required(VERSION 3.10)
project(common)

#为构建添加子目录
set(Common_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include CACHE PATH "Common_INCLUDE_DIRS")

execute_process(COMMAND ls ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE TEST_MODULE_COMMON_ST)
string(REPLACE "\n" ";" TEST_MODULE_COMMON_ST "${TEST_MODULE_COMMON_ST}")
foreach(m ${TEST_MODULE_COMMON_ST})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${m})
        add_subdirectory(${m})
    endif()
endforeach()

#指定可能用到的头文件路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/math/include)

file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/math/src/*.cpp)
add_library(common STATIC ${SRC})