cmake_minimum_required(VERSION 3.10)
project(rm3)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译优化选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 海康相机 SDK 路径
set(MVCAM_DIR /opt/MVS)

# 包含头文件目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${MVCAM_DIR}/include
)

# 链接库目录
link_directories(
    ${MVCAM_DIR}/lib/64
)

# 添加可执行文件
add_executable(rm3 main.cpp)

# 链接库
target_link_libraries(rm3
    ${OpenCV_LIBS}
    MvCameraControl
    pthread
)

# 安装规则（可选）
install(TARGETS rm3 DESTINATION bin)
