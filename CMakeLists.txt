cmake_minimum_required(VERSION 3.10)
project(ArmorDetection CXX)

# C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 海康相机 SDK 路径
set(HIKROBOT_SDK_ROOT /opt/MVS)

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${HIKROBOT_SDK_ROOT}/include)

# 链接库目录
link_directories(${HIKROBOT_SDK_ROOT}/lib/64)

# 可执行文件
add_executable(main rm2.cpp)

target_include_directories(main
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE ${HIKROBOT_SDK_ROOT}/include
)

target_link_libraries(main
    PRIVATE ${OpenCV_LIBS}
    PRIVATE MvCameraControl
    PRIVATE pthread
)
