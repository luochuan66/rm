# RM3 装甲板识别系统

## 项目结构

```
rm3/
├── algorithm.h    # 算法模块（灯条类、PnP位姿解算、卡尔曼滤波、小陀螺检测）
├── serial.h       # 串口通信模块
├── vision.h       # 图像处理模块（预处理、灯条提取、装甲板匹配、结果绘制）
├── main.cpp       # 主程序（相机初始化、主循环、数据处理）
├── CMakeLists.txt # CMake 构建配置
└── README.md      # 项目说明
```

## 模块说明

### 1. algorithm.h - 算法模块
- `LightDescriptor`: 灯条描述类
- `solveArmorPnP()`: PnP位姿解算
- `rvecToEuler()`: 旋转向量转欧拉角
- `ArmorKalmanFilter`: 卡尔曼滤波与小陀螺检测类

### 2. serial.h - 串口通信模块
- `SerialPort`: 串口通信类
- `VisionSendData`: 串口发送数据结构
- 支持自动打开、关闭、发送数据

### 3. vision.h - 图像处理模块
- `VisionProcessor`: 图像处理类
- `preprocess()`: 图像预处理（灰度、二值化、形态学）
- `extractLights()`: 提取灯条
- `matchArmor()`: 匹配装甲板
- `computePose()`: 计算姿态
- `drawResult()`: 绘制识别结果

### 4. main.cpp - 主程序
- 海康相机初始化配置
- 串口初始化
- 主处理循环
- 帧率统计

## 编译方法

### 使用 CMake 编译（推荐）

```bash
cd /home/luochuang/code/rm3
mkdir build && cd build
cmake ..
make
```

### 使用 g++ 直接编译

```bash
cd /home/luochuang/code/rm3
g++ -O3 -march=native main.cpp -o rm3 \
    -I../MVS_SDK/Linux64/inc \
    -L../MVS_SDK/Linux64/lib \
    -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d \
    -lMvCameraControl -lpthread -std=c++11
```

## 运行方法

```bash
cd /home/luochuang/code/rm3
./rm3
```

## 相机配置

- **相机**: 海康威视工业相机
- **分辨率**: 640 × 480
- **帧率**: 90 fps（自动降级到 60 fps 如果不支持）
- **曝光时间**: 2000 μs
- **增益**: 8.0

## 串口配置

- **端口**: /dev/ttyUSB0
- **波特率**: 115200
- **数据位**: 8
- **停止位**: 1
- **校验位**: 无

## 性能优化

1. **编译优化**: 使用 `-O3 -march=native` 选项
2. **图像缩放**: 处理时缩小到 50%
3. **容器预分配**: 减少动态内存分配
4. **减少 cout 输出**: 只在必要时输出调试信息

## 输出信息

### 每帧输出
```
检测中心: (571.53, 187.55)
距离: 0.34 m
Yaw: -107.45 度
预测置信度: 1.00
```

### 每秒输出
```
[FPS] 当前帧率: 90.0 fps
```

## 依赖项

- OpenCV >= 3.0
- 海康相机 SDK (MvCameraControl)
- Linux 系统

## 注意事项

1. 确保海康相机 SDK 路径正确
2. 确保串口设备存在且有权限
3. 相机支持 90fps 或 60fps
4. 曝光时间需要小于帧间隔（90fps ≤ 11.1ms）

## 延迟补偿参数

- 图像采集延迟: 11ms (90fps)
- 图像处理延迟: 5ms
- 串口传输延迟: 5ms
- 机械响应延迟: 50ms
- **总延迟**: 71ms
