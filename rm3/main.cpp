#include "algorithm.h"
#include "serial.h"
#include "vision.h"
#include "MvCameraControl.h"
#include <chrono>
#include <iomanip>

using namespace std;
using namespace cv;

// ===================== 延迟补偿参数 =====================
const float IMAGE_DELAY = 0.011f;       // 图像采集延迟 (90fps ≈ 11ms)
const float PROCESS_DELAY = 0.005f;      // 图像处理延迟 (5ms，优化后)
const float TRANSMISSION_DELAY = 0.005f; // 串口传输延迟 (5ms)
const float MECHANICAL_DELAY = 0.050f;  // 机械响应延迟 (50ms)
const float TOTAL_DELAY = IMAGE_DELAY + PROCESS_DELAY + TRANSMISSION_DELAY + MECHANICAL_DELAY; // 总延迟约71ms



// ===================== 主函数 =====================
int main()
{
    ArmorKalmanFilter armor_kalman(0.2f);//初始化卡尔曼滤波器
    bool is_kalman_init = false;//跟踪状态
    Point2f kalman_center;//// 卡尔曼滤波后的中心点
    Point2f last_armor_center(-1, -1);  // 记录上一帧装甲板中心
    int lost_frame_count = 0;//丢失检测机制
    const int MAX_LOST_FRAMES = 10;  // 增加最大丢失帧数

    // 连续性跟踪参数
    float last_valid_spacing = 0.0f;  // 上一帧有效间距
    float last_valid_distance = 0.0f;  // 上一帧有效距离
    int valid_detection_count = 0;  // 连续有效检测帧数
    const int MIN_VALID_FRAMES_FOR_TRACKING = 5;  // 最小连续检测帧数

    // ===================== 海康相机初始化 =====================
    unsigned int nTLayerType = MV_GIGE_DEVICE | MV_USB_DEVICE;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备
    int nRet = MV_CC_EnumDevices(nTLayerType, &stDeviceList);
    if (nRet != MV_OK || stDeviceList.nDeviceNum == 0) {
        cout << "未找到海康相机！" << endl;
        return -1;
    }

    cout << "找到 " << stDeviceList.nDeviceNum << " 个相机设备" << endl;

    // 选择第一个设备
    int nIndex = 0;
    MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[nIndex];//触发模式

    // 创建句柄
    void* handle = NULL;
    nRet = MV_CC_CreateHandle(&handle, pDeviceInfo);
    if (nRet != MV_OK) {
        cout << "创建相机句柄失败！" << endl;//句柄类似文件指针，后续所有操作都通过它进行。
        return -1;
    }

    // 打开设备
    nRet = MV_CC_OpenDevice(handle);
    if (nRet != MV_OK) {
        cout << "打开相机失败！" << endl;
        MV_CC_DestroyHandle(handle);
        return -1;
    }

    cout << "相机打开成功！" << endl;

    // 设置触发模式为关闭
    MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF);

    // 设置像素格式为 RGB8（彩色）
    MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_RGB8_Packed);

    // 设置曝光时间（90fps 需要 ≤ 11111μs，设置为 2000μs 以获得更好效果）2ms
    MV_CC_SetFloatValue(handle, "ExposureTime", 2000.0);

    // 设置增益（适当提高以补偿低曝光）
    MV_CC_SetFloatValue(handle, "Gain", 8.0);

    // 直接设置为 90 fps
    nRet = MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", 90.0);
    if (nRet == MV_OK) {
        cout << "已设置为 90.0 fps" << endl;
    } else {
        cout << "设置 90fps 失败！错误码: 0x" << hex << nRet << endl;
        // 如果 90fps 不支持，尝试设置 60fps
        nRet = MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", 60.0);
        if (nRet == MV_OK) {
            cout << "降级设置为 60.0 fps" << endl;
        } else {
            cout << "设置帧率失败，使用相机默认帧率" << endl;
        }
    }

    // 开始取流
    nRet = MV_CC_StartGrabbing(handle);
    if (nRet != MV_OK) {
        cout << "开始取流失败！" << endl;
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        return -1;
    }

    cout << "相机取流已启动！" << endl;

    // ===================== 初始化串口 =====================
    SerialPort serial;
    if (!serial.open("/dev/ttyUSB0")) {
        cout << "串口未连接，将不发送数据" << endl;
    }

    // ===================== 图像处理循环 =====================
    VisionProcessor vision;

    MV_FRAME_OUT_INFO_EX stImageInfo = {0};
    // 优化：根据实际分辨率调整缓冲区
    // 降低分辨率以提高帧率：使用640×480或更小
    unsigned int nDataSize = 640 * 480 * 3;  // 减小缓冲区以支持高帧率
    unsigned char* pData = (unsigned char*)malloc(nDataSize);

    // 设置更小的分辨率以提高帧率
    nRet = MV_CC_SetIntValue(handle, "Width", 640);
    if (nRet != MV_OK) {
        cout << "设置宽度失败，使用默认宽度" << endl;
    }
    nRet = MV_CC_SetIntValue(handle, "Height", 480);
    if (nRet != MV_OK) {
        cout << "设置高度失败，使用默认高度" << endl;
    }

    for (;;) {
        // 优化：减少memset调用，只在必要时清零结构体
        // 减少超时时间以快速响应，支持高帧率
        nRet = MV_CC_GetOneFrameTimeout(handle, pData, nDataSize, &stImageInfo, 10);//获取一帧
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
            // 降低输出频率，避免刷屏
            static int timeout_count = 0;
            if (++timeout_count % 100 == 0) {
                cout << "获取图像超时！错误码: 0x" << hex << nRet << endl;
            }
            continue;
        }

        // 帧率统计 静态变量初始化
        static int frame_counter = 0;// 帧计数器
        static auto fps_start_time = chrono::steady_clock::now();// 起始时间戳
        frame_counter++;
        auto current_time = chrono::steady_clock::now();// 记录当前时刻
        auto fps_elapsed = chrono::duration_cast<chrono::milliseconds>(current_time - fps_start_time).count();//计算时间差
        if (fps_elapsed >= 1000) {
            float fps = frame_counter * 1000.0f / fps_elapsed;
            cout << "[FPS] 当前帧率: " << fixed << setprecision(1) << fps << " fps" << endl;
            frame_counter = 0;
            fps_start_time = current_time;
        }

        // 转换为 OpenCV Mat（RGB格式）
        Mat rgb(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);//海康相机输出RGB格式，OpenCV默认使用BGR
        Mat frame;
        cvtColor(rgb, frame, COLOR_RGB2BGR);  // 转换为BGR格式供OpenCV显示

        if (frame.empty()) continue;

        // 优化：降低分辨率进行图像处理，提高帧率
        Mat frame_small;
        resize(frame, frame_small, Size(), 0.5, 0.5, INTER_LINEAR);  // 缩小到50%

        Mat frame_copy = frame_small.clone();  // 使用缩小后的图像进行处理

        // ===================== 图像预处理 =====================
        Mat binary = vision.preprocess(frame_small);

        // ===================== 提取灯条 =====================
        vector<LightDescriptor> lightInfos = vision.extractLights(binary);

        // ===================== 匹配装甲板 =====================
        Point2f armorCenter;//// 装甲板中心坐标（像素）
        vector<Point2f> vertices;//// 装甲板四个角点（用于PnP）
        float estimated_distance = 0.0f;//// 估计距离（米）
        bool armorFound = vision.matchArmor(lightInfos, armorCenter, vertices, estimated_distance,//核心检测函数调用
                                            last_armor_center, kalman_center, is_kalman_init,
                                            last_valid_spacing, last_valid_distance, valid_detection_count);

        if (armorFound) {//// 更新检测中心
            vision.detect_center = armorCenter;
            last_armor_center = armorCenter;  // 更新上一帧位置

            // 更新连续性跟踪参数 / 计算帧间位移（欧氏距离）
            last_valid_spacing = sqrt(pow(armorCenter.x - last_armor_center.x, 2) + pow(armorCenter.y - last_armor_center.y, 2));
            last_valid_distance = estimated_distance;
            valid_detection_count++;  // 增加连续检测帧数

            // ===================== 计算姿态 =====================
            if (vision.computePose(vertices, armorCenter)) {
                // ===================== 卡尔曼更新（先预测，后更新） =====================
                lost_frame_count = 0;
                if (!is_kalman_init) {
                    // 如果有上一帧位置，可以估计初始速度
                    const float dt = 0.2f;  // 与构造函数中的 dt 一致
                    if (last_armor_center.x > 0) {
                        Point2f init_velocity = (armorCenter - last_armor_center) / dt;
                        armor_kalman.initWithVelocity(armorCenter, init_velocity);
                        cout << "【初始化】使用估计速度: (" << init_velocity.x << ", " << init_velocity.y << ")" << endl;
                    } else {
                        armor_kalman.init(armorCenter);
                    }
                    is_kalman_init = true;
                    // 初始化后，直接使用检测点，不做预测
                    kalman_center = armorCenter;
                } else {
                    // 先更新（使用当前测量值更新卡尔曼状态）
                    armor_kalman.update(armorCenter);
                    // 再预测下一帧位置
                    bool is_spinning = armor_kalman.detectSpinning(armorCenter);
                    if (is_spinning) {
                        kalman_center = armor_kalman.adaptivePredict(armorCenter);
                    } else {
                        kalman_center = armor_kalman.predict();
                    }
                }

                // ===================== 延迟补偿预测（基于当前预测位置进行延迟补偿） =====================
                Point2f delay_compensated_pos = armor_kalman.predictWithDelayCompensation(TOTAL_DELAY);//延迟补偿预测
                Point3f delay_compensated_pose = armor_kalman.predictPoseWithDelay(TOTAL_DELAY, vision.pitch, vision.yaw);

                vision.prediction_confidence = armor_kalman.prediction_confidence;

                // 发送串口 - 使用延迟补偿后的数据
                if (serial.isOpen() && vision.distance_m > 0.5f && vision.distance_m < 10.0f) {
                    // 检测位置跳变（用于判断是否稳定）
                    static Point2f last_send_center(-1, -1);//稳定性判断机制
                    static int stable_frame_count = 0;
                    const int MIN_STABLE_FRAMES = 3;  // 至少稳定3帧才发送
                    const float MAX_JUMP_DISTANCE = 50.0f;  // 最大允许跳变距离

                    bool is_stable = true;//判断逻辑
                    if (last_send_center.x > 0) {
                        float jump_dist = sqrt(pow(armorCenter.x - last_send_center.x, 2) +
                                              pow(armorCenter.y - last_send_center.y, 2));
                        if (jump_dist > MAX_JUMP_DISTANCE) {
                            is_stable = false;
                            stable_frame_count = 0;
                        }
                    }

                    if (is_stable) {
                        stable_frame_count++;
                    }

                    // 只有稳定帧或卡尔曼置信度高时才发送
                    if (stable_frame_count >= MIN_STABLE_FRAMES || armor_kalman.prediction_confidence > 0.7f) {//发送条件判断（双重标准）
                        serial.sendVisionData(delay_compensated_pose.y,//数据发送
                                              vision.pitch + delay_compensated_pose.x * 0.3f,
                                              vision.distance_m,
                                              lost_frame_count == 0 && armor_kalman.prediction_confidence > 0.6f);
                        last_send_center = armorCenter;  // 更新上次发送位置
                    } else {
                        // 不稳定帧，跳过发送或发送上一次稳定的数据
                        static int skip_frame_count = 0;
                        if (++skip_frame_count % 50 == 0) {  // 减少输出频率
                            cout << "【跳帧】检测位置不稳定，跳过发送（跳变距离: "
                                 << sqrt(pow(armorCenter.x - last_send_center.x, 2) +
                                        pow(armorCenter.y - last_send_center.y, 2)) << "）" << endl;
                        }
                    }
                }

                // 每帧输出检测信息
                cout << "检测中心: (" << vision.detect_center.x << ", " << vision.detect_center.y << ")" << endl;
                cout << "距离: " << fixed << setprecision(2) << vision.distance_m << " m" << endl;
                cout << "Yaw: " << fixed << setprecision(2) << vision.yaw << " 度" << endl;
                cout << "预测置信度: " << fixed << setprecision(2) << vision.prediction_confidence << endl;

                // ===================== 绘制结果 =====================
                vision.drawResult(frame_copy, armorCenter, vertices, kalman_center, armor_kalman);

                // 绘制延迟补偿位置（黄色圆点）
                circle(frame_copy, delay_compensated_pos, 4, Scalar(0, 255, 255), -1);
                putText(frame_copy, "Delay", delay_compensated_pos + Point2f(10, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);

                // 绘制延迟轨迹线（从检测位置到补偿位置）
                line(frame_copy, armorCenter, delay_compensated_pos, Scalar(0, 255, 255), 1);
            }
        }

        // 卡尔曼逻辑（处理丢失帧的情况）
        if (!armorFound) {
            lost_frame_count++;
            valid_detection_count = 0;  // 重置连续检测帧数

            if (lost_frame_count > MAX_LOST_FRAMES) {//超时重置机制
                is_kalman_init = false;
                last_armor_center = Point2f(-1, -1);  // 重置上一帧位置
                last_valid_spacing = 0.0f;  // 重置有效间距
                last_valid_distance = 0.0f;  // 重置有效距离
                armor_kalman.reset();
                kalman_center = Point2f(-100, -100);
            } else if (is_kalman_init) {//预测跟踪模式
                // 丢失帧时仅预测，不更新
                bool is_spinning = armor_kalman.detectSpinning(kalman_center);
                if (is_spinning) {
                    kalman_center = armor_kalman.adaptivePredict(kalman_center);
                } else {
                    kalman_center = armor_kalman.predict();
                }
                // 限制在图像范围内//边界限制
                kalman_center.x = max(0.0f, min(kalman_center.x, (float)frame.cols));
                kalman_center.y = max(0.0f, min(kalman_center.y, (float)frame.rows));
            }
        }

        imshow("binary", binary);
        imshow("video", frame_copy);
        if (waitKey(1) == 27) break;
    }

    // 停止取流
    MV_CC_StopGrabbing(handle);

    // 关闭设备
    MV_CC_CloseDevice(handle);

    // 销毁句柄
    MV_CC_DestroyHandle(handle);

    // 释放图像缓冲区
    if (pData != NULL) {
        free(pData);
        pData = NULL;
    }

    // 最终清理：释放所有容器预留的内存
    vision.cleanup();

    serial.close();
    cv::destroyAllWindows();
    return 0;
}
