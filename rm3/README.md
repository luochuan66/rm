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
 // ===================== 评分系统总览 =====================
                //
                // 【评分项目及权重】
                // ┌─────────────────────────────────────────────────────────────────────────┐
                // │ 评分项                  │ 分数范围 │ 说明                                │
                // ├─────────────────────────────────────────────────────────────────────────┤
                // │ 1. 角度差评分          │ 0-25     │ 两灯条角度差，0°满分               │
                // │ 2. 长度一致性评分      │ 0-25     │ 两灯条长度一致性                    │
                // │ 3. 间距/长度比评分     │ 0-25     │ 灯条排列密度，1.5满分【距离评分1】  │
                // │ 4. Y轴间距评分         │ 0-25     │ Y轴对齐程度                         │
                // │ 5. 间距稳定性评分      │ 0-20     │ 与上一帧间距一致性【距离评分2】      │
                // │ 6. 距离稳定性评分      │ 0-10     │ 估算距离稳定性【距离评分3】        │
                // │ 7. 位置距离评分        │ 0-25/15  │ 与预测位置距离【距离评分4】        │
                // ├─────────────────────────────────────────────────────────────────────────┤
                // │ 总分（无跟踪）         │ 0-100    │ 1+2+3+4+7(15分)                    │
                // │ 总分（有跟踪）         │ 0-130    │ 1+2+3+4+5+6+7(25分)                │
                // └─────────────────────────────────────────────────────────────────────────┘
                //
                // 【调整评分权重的建议】
                // - 提高距离评分权重：增大评分5、6、7的分数系数
                // - 降低其他评分权重：减小评分1、2、3、4的分数系数
                // - 例如：将评分5从20改为30，评分7从25改为35
                //
                // 【距离相关评分】
                // - 评分3：ratio = dis/meanLen，反映灯条排列密度
                // - 评分5：spacing_change = |dis - last_valid_spacing| / last_valid_spacing
                // - 评分6：distance_change = |estimated_dist - last_valid_distance| / last_valid_distance
                // - 评分7：distToKalman = 装甲板中心到预测位置的距离
                //
                // 放宽几何特征约束，提高识别稳定性
                 // ===================== 评分系统 =====================
                // 总分范围：0-130分（有跟踪时）或 0-100分（无跟踪时）
                float score = 0;

                // 【评分1】角度差评分：0-25分
                // 角度差越小得分越高，0°时满分25分，40°时0分
                // 权重系数：25分
                score += (20 - angleGap_) / 20 * 25;  // 扩大角度容忍范围

                // 【评分2】长度一致性评分：0-25分
                // 两灯条长度完全一致时满分25分，长度差为meanLen时0分
                // 权重系数：25分
                score += (1 - lengap_ratio) * 25;

                // 【评分3】间距/长度比评分：0-25分（距离评分1）
                // ratio=1.5时满分25分（理想值），ratio=0或3时0分
                // 权重系数：25分
                // ratio反映了灯条排列密度，1.25为物理理论值（100mm间距/80mm长度）
                score += (1 - abs(ratio - 1.5)) * 25;

                // 【评分4】Y轴间距评分：0-25分
                // Y轴间距为0时满分25分，间距为2*meanLen时0分
                // 权重系数：25分
                score += (2.0 - yGap_ratio) / 2.0 * 25;  // 扩大Y轴差距容忍范围

                // ===================== 连续性跟踪加分 =====================
                // 仅当有效帧数 >= 5 且有上一次有效间距时启用
                if (use_tracking_bonus && last_valid_spacing > 0) {
                    float spacing_change = abs(dis - last_valid_spacing) / last_valid_spacing;
                    float distance_change = abs(estimated_dist - last_valid_distance) / last_valid_distance;

                    // 【评分5】间距稳定性评分：0-20分（距离评分2）
                    // 与上一帧间距完全一致时满分20分，间距变化100%时0分
                    // 权重系数：20分
                    score += max(0.0f, 1.0f - spacing_change) * 20;  // 间距稳定性

                    // 【评分6】距离稳定性评分：0-10分（距离评分3）
                    // 与上一帧距离完全一致时满分10分，距离变化100%时0分
                    // 权重系数：10分
                    score += max(0.0f, 1.0f - distance_change) * 10;  // 距离稳定性
                }

                // ===================== 位置距离评分 =====================
                // 添加上一帧位置权重，根据连续性调整权重
                float position_weight = use_tracking_bonus ? 25 : 15;  // 有跟踪时增加权重：15-25分
                float position_range = use_tracking_bonus ? 200 : 150;  // 有跟踪时扩大范围：150-200px

                if (is_kalman_init && kalman_center.x > 0) {
                    Point2f armorC = Point2f((leftLight.center.x + rightLight.center.x) / 2,
                                           (leftLight.center.y + rightLight.center.y) / 2);
                    float distToKalman = sqrt(pow(armorC.x - kalman_center.x, 2) +
                                            pow(armorC.y - kalman_center.y, 2));

                    // 【评分7】卡尔曼预测位置距离评分：0-25分（有跟踪）或 0-15分（无跟踪）（距离评分4）
                    // 距离卡尔曼预测位置0px时满分，距离=position_range时0分
                    // 权重系数：position_weight（15-25分）
                    score += max(0.0f, position_range - distToKalman) / position_range * position_weight;
                } else if (last_armor_center.x > 0) {
                    Point2f armorC = Point2f((leftLight.center.x + rightLight.center.x) / 2,
                                           (leftLight.center.y + rightLight.center.y) / 2);
                    float distToLast = sqrt(pow(armorC.x - last_armor_center.x, 2) +
                                           pow(armorC.y - last_armor_center.y, 2));

                    // 【评分7替代】上一帧位置距离评分：0-25分（有跟踪）或 0-15分（无跟踪）（距离评分4）
                    // 距离上一帧位置0px时满分，距离=position_range时0分
                    // 权重系数：position_weight（15-25分）
