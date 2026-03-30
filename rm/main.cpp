// ===================== 标准库 =====================
#include <iostream>
#include <iomanip>
#include <chrono>

// ===================== 自定义头文件（先包含海康SDK） =====================
#include "camera_control.h"  // 必须最先包含，因为它包含MvCameraControl.h
#include "armor_detector.h"
#include "kalman_filter.h"
#include "serial_port.h"

// ===================== OpenCV =====================
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// ===================== HSV阈值 =====================
int H_min = 72, H_max = 101;   // 蓝色色
int S_min = 37, S_max = 255;
int V_min = 149, V_max = 255;

// ===================== 延迟补偿参数 =====================
const float IMAGE_DELAY = 0.033f;       // 图像采集延迟 (30fps ≈ 33ms)
const float PROCESS_DELAY = 0.010f;      // 图像处理延迟 (10ms)
const float TRANSMISSION_DELAY = 0.005f; // 串口传输延迟 (5ms)
const float MECHANICAL_DELAY = 0.050f;  // 机械响应延迟 (50ms)
const float TOTAL_DELAY = IMAGE_DELAY + PROCESS_DELAY + TRANSMISSION_DELAY + MECHANICAL_DELAY; // 总延迟约98ms

// ===================== 滑块回调 =====================
void onTrackbar(int, void*) {}

int main() {
    ArmorKalmanFilter armor_kalman(0.2f);
    bool is_kalman_init = false;
    Point2f kalman_center;
    Point2f last_armor_center(-1, -1);  // 记录上一帧装甲板中心
    int lost_frame_count = 0;
    const int MAX_LOST_FRAMES = 10;  // 增加最大丢失帧数

    // 连续性跟踪参数
    float last_valid_spacing = 0.0f;  // 上一帧有效间距
    float last_valid_distance = 0.0f;  // 上一帧有效距离
    int valid_detection_count = 0;  // 连续有效检测帧数
    const int MIN_VALID_FRAMES_FOR_TRACKING = 5;  // 最小连续检测帧数

    // ===================== 海康相机初始化 =====================
    HikCamera camera;
    if (!camera.init()) {
        return -1;
    }

    if (!camera.startGrabbing()) {
        camera.release();
        return -1;
    }

    // ===================== 初始化串口 =====================
    SerialPort serial;
    serial.open("/dev/ttyUSB0", 115200);

    // ===================== 创建并调整窗口大小 =====================
    namedWindow("video", WINDOW_NORMAL);
    namedWindow("binary", WINDOW_NORMAL);
    resizeWindow("video", 1280, 720);   // 视频窗口放大到 1280x720
    resizeWindow("binary", 640, 480);   // 二值图窗口 640x480

    // ===================== 图像处理循环 =====================
    Mat element_small = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat element_medium = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat element_large = getStructuringElement(MORPH_RECT, Size(9, 9));
    Mat element_huge = getStructuringElement(MORPH_RECT, Size(15, 15));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // 优化：限制轮廓缓存为30，减少内存占用
    contours.reserve(30);
    hierarchy.reserve(30);

    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    // 优化：使用640×480分辨率计算缓冲区大小
    unsigned int nDataSize = 640 * 480 * 3;  // 640×480×3 RGB ≈ 921KB
    unsigned char* pData = (unsigned char*)malloc(nDataSize);

    // 预分配图像处理缓冲区（避免每帧重新分配）
    Mat frame(480, 640, CV_8UC3);
    Mat gray(480, 640, CV_8UC1);
    Mat binary(480, 640, CV_8UC1);
    Mat frame_copy(480, 640, CV_8UC3);
    Mat dilatee(480, 640, CV_8UC1);
    Mat morph(480, 640, CV_8UC1);

    // 帧率计算变量
    int frame_count = 0;
    auto last_time = chrono::steady_clock::now();
    float fps = 0.0f;

    for (;;) {
        // 计算帧率
        frame_count++;
        auto current_time = chrono::steady_clock::now();
        auto time_diff = chrono::duration_cast<chrono::milliseconds>(current_time - last_time).count();

        if (time_diff >= 1000) {
            fps = frame_count * 1000.0f / time_diff;
            frame_count = 0;
            last_time = current_time;
        }

        // 优化：减少memset调用，只在必要时清零结构体
        int nRet = camera.getOneFrame(pData, nDataSize, &stImageInfo, 1000);
        if (nRet != MV_OK) {
            // 如果是缓冲区不足错误，尝试扩大缓冲区
            if (nRet == MV_E_NODATA || stImageInfo.nFrameLen > nDataSize) {
                if (stImageInfo.nFrameLen > nDataSize) {
                    cout << "[缓冲区] 重新分配: " << nDataSize << " -> " << stImageInfo.nFrameLen << endl;
                    free(pData);
                    nDataSize = stImageInfo.nFrameLen * 2;
                    pData = (unsigned char*)malloc(nDataSize);
                    continue;
                }
            }
            cout << "获取图像超时！错误码: 0x" << hex << nRet << endl;
            continue;
        }

        // 转换为 OpenCV Mat（RGB格式）
        Mat rgb(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
        rgb.copyTo(frame);  // 使用预分配的frame，避免clone()
        cvtColor(rgb, frame, COLOR_RGB2BGR);  // 转换为BGR格式供OpenCV显示

        if (frame.empty()) continue;
        frame.copyTo(frame_copy);  // 使用预分配的frame_copy
        bool is_armor_detected = false;
        Point2f detect_center;

        // 转灰度图（使用预分配的gray）
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 降低阈值以识别红色灯条（80-255）
        threshold(gray, binary, 120, 255, THRESH_BINARY);

     
        // 高斯模糊
        GaussianBlur(binary, binary, Size(5, 5), 1.0);

        // 形态学操作：开运算去噪 + 闭运算填充
        morphologyEx(binary, morph, MORPH_OPEN, element_medium);
        morphologyEx(morph, morph, MORPH_CLOSE, element_medium);
        morph.copyTo(binary);  // 复制回binary

        // 膨胀连接断裂区域
        dilate(binary, dilatee, element_small, Point(-1, -1), 2);
        dilate(dilatee, binary, element_medium, Point(-1, -1), 1);

        // 查找轮廓
        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        // 优化：预分配lightInfos容量
        vector<LightDescriptor> lightInfos;
        lightInfos.reserve(contours.size() / 4);  // 估计约1/4的轮廓是有效灯条

        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 80 || contours[i].size() <= 10) continue;

            RotatedRect Light_Rec = fitEllipse(contours[i]);

            // 计算长宽比（长边/短边）
            float aspect_ratio = max(Light_Rec.size.width, Light_Rec.size.height) /
                                 min(Light_Rec.size.width, Light_Rec.size.height);
            // 针对中距离优化：长宽比 < 1.8 或 > 8 的轮廓
            if (aspect_ratio < 3 || aspect_ratio > 8) continue;

            // 针对中距离优化：面积过滤更宽松
            if (Light_Rec.size.area() < 50) continue;

            lightInfos.push_back(LightDescriptor(Light_Rec));
        }

        vector<bool> lightUsed(lightInfos.size(), false);
        bool armorFound = false;
        Point2f bestArmorCenter;
        float bestArmorScore = -1;
        int bestLightI = -1, bestLightJ = -1;

        for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; j < lightInfos.size(); j++) {
                LightDescriptor& leftLight = lightInfos[i];
                LightDescriptor& rightLight = lightInfos[j];
                float angleGap_ = abs(leftLight.angle - rightLight.angle);
                float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
                float dis = sqrt(pow(leftLight.center.x - rightLight.center.x, 2) + pow(leftLight.center.y - rightLight.center.y, 2));
                float meanLen = (leftLight.length + rightLight.length) / 2;
                float lengap_ratio = abs(leftLight.length - rightLight.length) / meanLen;
                float yGap_ratio = abs(leftLight.center.y - rightLight.center.y) / meanLen;
                float xGap_ratio = abs(leftLight.center.x - rightLight.center.x) / meanLen;
                float ratio = dis / meanLen;

                // ===================== 基于物理约束的灯条间距过滤（10cm） =====================
                // 根据灯条长度反推距离，判断灯条间距是否符合物理约束
                const float LIGHT_BAR_PHYSICAL_LENGTH = 80.0f;  // 灯条物理长度约80mm
                const float LIGHT_BAR_SPACING = 100.0f;  // 灯条间距约100mm

                float estimated_distance = (LIGHT_BAR_PHYSICAL_LENGTH * 500.0f) / meanLen;  // 估计距离(mm)
                float theoretical_spacing_px = (LIGHT_BAR_SPACING * 500.0f) / estimated_distance;  // 理论间距(像素)
                float spacing_error_ratio = abs(dis - theoretical_spacing_px) / theoretical_spacing_px;  // 间距误差比

                // 根据连续性跟踪调整间距误差容忍度
                float max_spacing_error;
                bool use_tracking_bonus = (valid_detection_count >= MIN_VALID_FRAMES_FOR_TRACKING) &&
                                         (last_valid_spacing > 0);

                // 针对主要识别距离 2.2m 优化
                if (estimated_distance > 2500.0f) {
                    // 远距离(>2.5m)
                    max_spacing_error = use_tracking_bonus ? 0.80 : 0.65;
                } else if (estimated_distance > 1800.0f) {
                    // 中远距离(1.8-2.5m) - 主要识别范围
                    max_spacing_error = use_tracking_bonus ? 0.90 : 0.75;
                } else if (estimated_distance > 1200.0f) {
                    // 中距离(1.2-1.8m)
                    max_spacing_error = use_tracking_bonus ? 1.00 : 0.85;
                } else {
                    // 近距离(≤1.2m)
                    max_spacing_error = use_tracking_bonus ? 1.10 : 0.95;
                }

                // 关键过滤：灯条间距必须符合物理约束
                if (spacing_error_ratio > max_spacing_error) continue;

                // 放宽几何特征约束，提高识别稳定性
                if (angleGap_ > 40 || LenGap_ratio > 2.0 || lengap_ratio > 1.5 ||
                    yGap_ratio > 3.0 || xGap_ratio > 4.0 || xGap_ratio < 0.5 ||
                    ratio > 4 || ratio < 1.2) continue;

                float score = 0;
                score += (20 - angleGap_) / 20 * 25;  // 扩大角度容忍范围
                score += (1 - lengap_ratio) * 25;
                score += (1 - abs(ratio - 1.5)) * 25;
                score += (2.0 - yGap_ratio) / 2.0 * 25;  // 扩大Y轴差距容忍范围

                // 连续性跟踪：间距和距离的稳定性权重
                if (use_tracking_bonus && last_valid_spacing > 0) {
                    float spacing_change = abs(dis - last_valid_spacing) / last_valid_spacing;
                    float distance_change = abs(estimated_distance - last_valid_distance) / last_valid_distance;
                    score += max(0.0f, 1.0f - spacing_change) * 20;  // 间距稳定性
                    score += max(0.0f, 1.0f - distance_change) * 10;  // 距离稳定性
                }

                // 添加上一帧位置权重，根据连续性调整权重
                float position_weight = use_tracking_bonus ? 25 : 15;  // 有跟踪时增加权重
                float position_range = use_tracking_bonus ? 200 : 150;  // 有跟踪时扩大范围

                if (is_kalman_init && kalman_center.x > 0) {
                    Point2f armorCenter = Point2f((leftLight.center.x + rightLight.center.x) / 2,
                                                   (leftLight.center.y + rightLight.center.y) / 2);
                    float distToKalman = sqrt(pow(armorCenter.x - kalman_center.x, 2) +
                                            pow(armorCenter.y - kalman_center.y, 2));
                    score += max(0.0f, position_range - distToKalman) / position_range * position_weight;
                } else if (last_armor_center.x > 0) {
                    Point2f armorCenter = Point2f((leftLight.center.x + rightLight.center.x) / 2,
                                                   (leftLight.center.y + rightLight.center.y) / 2);
                    float distToLast = sqrt(pow(armorCenter.x - last_armor_center.x, 2) +
                                           pow(armorCenter.y - last_armor_center.y, 2));
                    score += max(0.0f, position_range - distToLast) / position_range * position_weight;
                }

                if (score > bestArmorScore) {
                    bestArmorScore = score;
                    bestArmorCenter = Point2f((leftLight.center.x + rightLight.center.x) / 2,
                                             (leftLight.center.y + rightLight.center.y) / 2);
                    bestLightI = i;
                    bestLightJ = j;
                    armorFound = true;
                }
            }
        }

        if (armorFound) {
            lightUsed[bestLightI] = true;
            lightUsed[bestLightJ] = true;
            detect_center = bestArmorCenter;
            is_armor_detected = true;
            last_armor_center = detect_center;  // 更新上一帧位置

            LightDescriptor& leftLight = lightInfos[bestLightI];
            LightDescriptor& rightLight = lightInfos[bestLightJ];
            float dis = sqrt(pow(leftLight.center.x - rightLight.center.x, 2) + pow(leftLight.center.y - rightLight.center.y, 2));
            float meanLen = (leftLight.length + rightLight.length) / 2;

            // 更新连续性跟踪参数
            const float LIGHT_BAR_PHYSICAL_LENGTH = 80.0f;
            float estimated_distance = (LIGHT_BAR_PHYSICAL_LENGTH * 500.0f) / meanLen;
            last_valid_spacing = dis;
            last_valid_distance = estimated_distance;
            valid_detection_count++;  // 增加连续检测帧数

            RotatedRect rect = RotatedRect(detect_center, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
            Point2f vertices[4];
            rect.points(vertices);
            vector<Point2f> vertexVec(vertices, vertices + 4);

            for (int k = 0; k < 4; k++) line(frame_copy, vertices[k], vertices[(k + 1) % 4], Scalar(0, 0, 255), 2.2);
            circle(frame_copy, detect_center, 4, Scalar(255, 0, 0), -1);
            putText(frame_copy, "Detect", detect_center + Point2f(10, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);

            Mat rvec, tvec;
            if (solveArmorPnP(vertexVec, rvec, tvec)) {
                float distance = (float)tvec.at<double>(2);
                float distance_m = distance / 1000.0f;
                Vec3f euler = rvecToEuler(rvec);
                float pitch = euler[0];
                float yaw = euler[1];

                float x = tvec.at<double>(0) / 1000.0f;
                // float y = tvec.at<double>(1) / 1000.0f;  // 保留供将来使用
                float z = tvec.at<double>(2) / 1000.0f;
                float distance_real = sqrt(x*x + z*z);

                // ===================== 卡尔曼更新（先预测，后更新） =====================
                lost_frame_count = 0;
                if (!is_kalman_init) {
                    // 如果有上一帧位置，可以估计初始速度
                    const float dt = 0.2f;  // 与构造函数中的 dt 一致
                    if (last_armor_center.x > 0) {
                        Point2f init_velocity = (detect_center - last_armor_center) / dt;
                        armor_kalman.initWithVelocity(detect_center, init_velocity);
                        cout << "【初始化】使用估计速度: (" << init_velocity.x << ", " << init_velocity.y << ")" << endl;
                    } else {
                        armor_kalman.init(detect_center);
                    }
                    is_kalman_init = true;
                    // 初始化后，直接使用检测点，不做预测
                    kalman_center = detect_center;
                } else {
                    // 先更新（使用当前测量值更新卡尔曼状态）
                    armor_kalman.update(detect_center);
                    // 再预测下一帧位置
                    bool is_spinning = armor_kalman.detectSpinning(detect_center);
                    if (is_spinning) {
                        kalman_center = armor_kalman.adaptivePredict(detect_center);
                    } else {
                        kalman_center = armor_kalman.predict();
                    }
                }

                // ===================== 延迟补偿预测（基于当前预测位置进行延迟补偿） =====================
                Point2f delay_compensated_pos = armor_kalman.predictWithDelayCompensation(TOTAL_DELAY);
                Point3f delay_compensated_pose = armor_kalman.predictPoseWithDelay(TOTAL_DELAY, pitch, yaw);

                // 发送串口 - 使用延迟补偿后的数据
                if (serial.isOpen() && distance_real > 0.5f && distance_real < 10.0f) {
                    VisionSendData send_data;
                    send_data.header = 0xA5;

                    // 检测位置跳变（用于判断是否稳定）
                    static Point2f last_send_center(-1, -1);
                    static int stable_frame_count = 0;
                    const int MIN_STABLE_FRAMES = 3;  // 至少稳定3帧才发送
                    const float MAX_JUMP_DISTANCE = 50.0f;  // 最大允许跳变距离

                    bool is_stable = true;
                    if (last_send_center.x > 0) {
                        float jump_dist = sqrt(pow(detect_center.x - last_send_center.x, 2) +
                                              pow(detect_center.y - last_send_center.y, 2));
                        if (jump_dist > MAX_JUMP_DISTANCE) {
                            is_stable = false;
                            stable_frame_count = 0;
                        }
                    }

                    if (is_stable) {
                        stable_frame_count++;
                    }

                    // 只有稳定帧或卡尔曼置信度高时才发送
                    if (stable_frame_count >= MIN_STABLE_FRAMES || armor_kalman.prediction_confidence > 0.7f) {
                        send_data.yaw = delay_compensated_pose.y;      // 使用延迟补偿后的yaw
                        send_data.pitch = pitch + delay_compensated_pose.x * 0.3f;  // 考虑延迟补偿的pitch调整
                        send_data.distance = distance_real;
                        send_data.shoot = (lost_frame_count == 0 && armor_kalman.prediction_confidence > 0.6f) ? 1 : 0;  // 高置信度才射击
                        send_data.checksum = SerialPort::checkSum((uint8_t*)&send_data, sizeof(send_data)-1);
                        int n = serial.writeData(&send_data, sizeof(send_data));
                        if (n < 0) perror("串口写入失败");

                        last_send_center = detect_center;  // 更新上次发送位置
                    } else {
                        // 不稳定帧，跳过发送或发送上一次稳定的数据
                        cout << "【跳帧】检测位置不稳定，跳过发送（跳变距离: "
                             << sqrt(pow(detect_center.x - last_send_center.x, 2) +
                                    pow(detect_center.y - last_send_center.y, 2)) << "）" << endl;
                    }
                }

                cout << "========================================" << endl;
                cout << "检测中心: (" << detect_center.x << ", " << detect_center.y << ")" << endl;
                cout << "卡尔曼预测中心: (" << kalman_center.x << ", " << kalman_center.y << ")" << endl;
                cout << "距离: " << fixed << setprecision(2) << distance_m << " m (" << distance << " mm)" << endl;
                cout << "仰角(pitch): " << fixed << setprecision(2) << pitch << " 度" << endl;
                cout << "摆角(yaw): " << fixed << setprecision(2) << yaw << " 度" << endl;
                cout << "延迟补偿后Yaw: " << fixed << setprecision(2) << delay_compensated_pose.y << " 度" << endl;
                cout << "延迟补偿后位置: (" << fixed << setprecision(2) << delay_compensated_pos.x << ", "
                     << delay_compensated_pos.y << ")" << endl;
                cout << "总延迟时间: " << fixed << setprecision(3) << TOTAL_DELAY * 1000 << " ms" << endl;

                // 小陀螺状态信息
                cout << "角速度: " << fixed << setprecision(2) << armor_kalman.angular_velocity << " 度/帧" << endl;
                cout << "旋转状态: " << (armor_kalman.is_spinning ? "是" : "否") << endl;
                cout << "预测模式: " << (armor_kalman.use_adaptive_predict ? "自适应" : "标准") << endl;
                cout << "预测置信度: " << fixed << setprecision(2) << armor_kalman.prediction_confidence << endl;
                cout << "========================================" << endl;

                string distText = "Dist: " + to_string(static_cast<int>(distance_m * 10) / 10.0f) + "m";
                putText(frame_copy, distText, detect_center + Point2f(10, 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

                // 绘制延迟补偿位置（黄色圆点）
                circle(frame_copy, delay_compensated_pos, 4, Scalar(0, 255, 255), -1);
                putText(frame_copy, "Delay", delay_compensated_pos + Point2f(10, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);

                // 绘制延迟轨迹线（从检测位置到补偿位置）
                line(frame_copy, detect_center, delay_compensated_pos, Scalar(0, 255, 255), 1);
            }
        }

        // 卡尔曼逻辑（处理丢失帧的情况）
        if (!is_armor_detected) {
            lost_frame_count++;
            valid_detection_count = 0;  // 重置连续检测帧数

            if (lost_frame_count > MAX_LOST_FRAMES) {
                is_kalman_init = false;
                last_armor_center = Point2f(-1, -1);  // 重置上一帧位置
                last_valid_spacing = 0.0f;  // 重置有效间距
                last_valid_distance = 0.0f;  // 重置有效距离
                armor_kalman.reset();
                kalman_center = Point2f(-100, -100);
            } else if (is_kalman_init) {
                // 丢失帧时仅预测，不更新
                bool is_spinning = armor_kalman.detectSpinning(kalman_center);
                if (is_spinning) {
                    kalman_center = armor_kalman.adaptivePredict(kalman_center);
                } else {
                    kalman_center = armor_kalman.predict();
                }
                // 限制在图像范围内
                kalman_center.x = max(0.0f, min(kalman_center.x, (float)frame.cols));
                kalman_center.y = max(0.0f, min(kalman_center.y, (float)frame.rows));
            }
        }

        // 优化：每帧清理轮廓和层级容器，保持小内存占用
        contours.clear();
        hierarchy.clear();
        contours.shrink_to_fit();
        hierarchy.shrink_to_fit();

        if (is_kalman_init && kalman_center.x > 0 && kalman_center.y > 0) {
            circle(frame_copy, kalman_center, 4, Scalar(0, 0, 255), -1);

            string predText = armor_kalman.use_adaptive_predict ? "Adaptive" : "Kalman";
            putText(frame_copy, predText, kalman_center + Point2f(10, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

            // 小陀螺状态显示
            if (armor_kalman.is_spinning) {
                string spinText = "SPIN: " + to_string(static_cast<int>(armor_kalman.angular_velocity));
                putText(frame_copy, spinText, kalman_center + Point2f(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            }
        }

        // 在左上角显示实时帧率
        string fpsText = "FPS: " + to_string(static_cast<int>(fps));
        putText(frame_copy, fpsText, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

        imshow("binary", binary);
        imshow("video", frame_copy);
        if (waitKey(1) == 27) break;
    }

    // 释放相机资源
    camera.release();

    // 释放图像缓冲区
    if (pData != NULL) {
        free(pData);
        pData = NULL;
    }

    // 最终清理：释放所有容器预留的内存
    contours.clear();
    contours.shrink_to_fit();
    hierarchy.clear();
    hierarchy.shrink_to_fit();

    serial.close();
    cv::destroyAllWindows();
    return 0;
}
