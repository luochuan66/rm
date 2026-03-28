cmake_minimum_required(VERSION 3.10)
project(ArmorDetection CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 如果找不到 OpenCV，给出友好提示
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV 未找到！请先安装 OpenCV:\n"
                       "  Ubuntu/Debian: sudo apt install libopencv-dev\n"
                       "  CentOS/RHEL: sudo yum install opencv-devel\n"
                       "  或从源码编译: https://opencv.org/")
endif()

message(STATUS "OpenCV 版本: ${OpenCV_VERSION}")
message(STATUS "OpenCV 包含目录: ${OpenCV_INCLUDE_DIRS}")

# 添加可执行文件
add_executable(armor_detection gpt13b.cpp)

# 包含 OpenCV 头文件
target_include_directories(armor_detection PRIVATE ${OpenCV_INCLUDE_DIRS})

# 链接 OpenCV 库
target_link_libraries(armor_detection ${OpenCV_LIBS})

# 如果是 Debug 模式，添加调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(armor_detection PRIVATE -g -O0)
    message(STATUS "编译模式: Debug (带调试信息)")
else()
    target_compile_options(armor_detection PRIVATE -O2)
    message(STATUS "编译模式: Release (优化)")
endif()

message(STATUS "========================================")
message(STATUS "编译说明:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")
message(STATUS "  ./armor_detection")
message(STATUS "========================================")
