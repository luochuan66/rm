# 设置最低版本
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(rm2)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 设置海康SDK路径（统一命名规范）
set(HIKROBOT_SDK_ROOT /opt/MVS)

# 设置源文件（注意：源文件应该包含.cpp扩展名）
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/armor_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kalman_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/camera_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/serial_port.cpp
)

# 包含头文件目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${HIKROBOT_SDK_ROOT}/include
)

# 链接库目录（使用正确的库路径 lib/64）
link_directories(${HIKROBOT_SDK_ROOT}/lib/64)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库（注意语法规范）
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${HIKROBOT_SDK_ROOT}/lib/64/libMvCameraControl.so
    pthread
)

# 编译选项
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O2)
