cmake_minimum_required(VERSION 3.10)
project(ArmorDetection CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 可执行文件
add_executable(main rm.cpp)

target_include_directories(main
    PRIVATE ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(main
    PRIVATE ${OpenCV_LIBS}
)
