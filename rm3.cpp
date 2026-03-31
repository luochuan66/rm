#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>  // PnP算法
#include <cmath>  // 卡尔曼滤波需要的数学库
#include <iomanip>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// 海康相机 SDK
#include "MvCameraControl.h"

using namespace std;
using namespace cv;

// ===================== 装甲板物理尺寸（单位：mm） =====================
const float ARMOR_WIDTH = 140.0f;    
const float ARMOR_HEIGHT = 125.0f;   

// ===================== 相机内参（标定结果） =====================
const Mat CAMERA_MATRIX = (Mat_<double>(3, 3) <<
    589.0427686678858, 0.0, 1041.580514579736,
    0.0, 589.3795612894731, 534.8622665285162,
    0.0, 0.0, 1.0);
const Mat DIST_COEFFS = (Mat_<double>(5, 1) <<
    0.002445382552691366, -0.01440570673944744, -0.0001341828334394887,
    -0.0002228839280500997, -0.0006838812197758657);

// ===================== HSV阈值 =====================

int H_min = 72, H_max = 101;   //蓝色色
int S_min = 37, S_max = 255;
int V_min = 149, V_max = 255;

// ===================== 延迟补偿参数 =====================
const float IMAGE_DELAY = 0.011f;       // 图像采集延迟 (90fps ≈ 11ms)
const float PROCESS_DELAY = 0.005f;      // 图像处理延迟 (5ms，优化后)
const float TRANSMISSION_DELAY = 0.005f; // 串口传输延迟 (5ms)
const float MECHANICAL_DELAY = 0.050f;  // 机械响应延迟 (50ms)
const float TOTAL_DELAY = IMAGE_DELAY + PROCESS_DELAY + TRANSMISSION_DELAY + MECHANICAL_DELAY; // 总延迟约71ms

// ===================== 灯条类 =====================
class LightDescriptor
{	   
public:
    float width, length, angle, area;
    cv::Point2f center;
public:
    LightDescriptor() {};
    LightDescriptor(const cv::RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};

// ===================== PnP位姿解算 =====================
bool solveArmorPnP(const vector<Point2f>& vertices, Mat& rvec, Mat& tvec) {
    vector<Point3f> objectPoints = {
        Point3f(-ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0),
        Point3f(ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0),
        Point3f(ARMOR_WIDTH/2, ARMOR_HEIGHT/2, 0),
        Point3f(-ARMOR_WIDTH/2, ARMOR_HEIGHT/2, 0)
    };
    return solvePnP(objectPoints, vertices, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, false, SOLVEPNP_ITERATIVE);
}

Vec3f rvecToEuler(const Mat& rvec) {
    Mat R;
    Rodrigues(rvec, R);
    float yaw = atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * 180.0f / CV_PI;
    float pitch = atan2(-R.at<double>(2, 0), sqrt(R.at<double>(2, 1) * R.at<double>(2, 1) +
                                                      R.at<double>(2, 2) * R.at<double>(2, 2))) * 180.0f / CV_PI;
    float roll = atan2(R.at<double>(2, 1), R.at<double>(2, 2)) * 180.0f / CV_PI;
    return Vec3f(pitch, yaw, roll);
}

// ===================== 卡尔曼滤波与小陀螺检测 =====================
class ArmorKalmanFilter {
private:
    KalmanFilter kf;
    Mat state;
    Mat measurement;
    Point2f last_valid_pos;
    bool has_last_pos;

private:
    // 小陀螺检测相关
    vector<Point2f> trajectory;
    const int TRAJECTORY_LENGTH = 30;
    int spin_frame_count;
    const int SPIN_CONFIRM_FRAMES = 15;
    const float SPIN_ANGULAR_THRESHOLD = 30.0f;  // 角速度阈值(度/帧)

    float adapt_factor;
    float dt;

public:
    // 公开成员变量（用于显示和调试）
    float angular_velocity;
    bool is_spinning;
    bool use_adaptive_predict;
    float prediction_confidence;

    ArmorKalmanFilter(float dt_ = 0.2f) : dt(dt_), has_last_pos(false),
                                          angular_velocity(0.0f), is_spinning(false), spin_frame_count(0),
                                          use_adaptive_predict(false), adapt_factor(0.5f), prediction_confidence(1.0f) {
        kf = KalmanFilter(4, 2, 0);
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1);
        kf.measurementMatrix = (Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);
        // 激进预测：增大过程噪声，允许更大的速度变化
        kf.processNoiseCov = (Mat_<float>(4, 4) <<
            0.5, 0, 0, 0,
            0, 0.5, 0, 0,
            0, 0, 5.0, 0,  // 增大速度噪声
            0, 0, 0, 5.0);  // 增大速度噪声
        // 减小测量噪声，更多相信预测而非测量
        kf.measurementNoiseCov = (Mat_<float>(2, 2) <<
            0.05, 0,
            0, 0.05);
        setIdentity(kf.errorCovPost, Scalar::all(1));
        state = Mat::zeros(4, 1, CV_32F);
        measurement = Mat::zeros(2, 1, CV_32F);
    }

    // ===================== 延迟补偿预测 =====================
    Point2f predictWithDelayCompensation(float delay_time) {
        // 根据总延迟时间进行多步预测
        int prediction_steps = static_cast<int>(delay_time / dt);
        prediction_steps = max(1, min(prediction_steps, 15));  // 增加到15步，允许更远预测

        state = kf.predict();  // 初始预测

        for (int i = 1; i < prediction_steps; i++) {
            // 手动推进状态，速度不衰减
            float vx = state.at<float>(2);
            float vy = state.at<float>(3);
            state.at<float>(0) += vx * dt;
            state.at<float>(1) += vy * dt;
            // 保持原速度，不做任何衰减，激进预测
        }

        // 激进预测：大幅增加允许的移动距离
        if (has_last_pos) {
            float max_move = 300.0f;  // 增大到300像素，允许激进延迟补偿
            float dx = state.at<float>(0) - last_valid_pos.x;
            float dy = state.at<float>(1) - last_valid_pos.y;
            float dist = sqrt(dx*dx + dy*dy);
            float ratio = 1.0f;

            if (dist > max_move) {
                ratio = max_move / dist;
                state.at<float>(0) = last_valid_pos.x + dx * ratio;
                state.at<float>(1) = last_valid_pos.y + dy * ratio;
            }

            kf.statePost = state;
        }

        return Point2f(state.at<float>(0), state.at<float>(1));
    }

    // ===================== 带延迟的姿态预测 =====================
    Point3f predictPoseWithDelay(float delay_time, float current_pitch, float current_yaw) {
        // 预测延迟后的pitch和yaw
        int prediction_steps = static_cast<int>(delay_time / dt);
        prediction_steps = max(1, min(prediction_steps, 10));

        // 获取速度
        float vx = state.at<float>(2);
        float vy = state.at<float>(3);

        // 预测位置变化
        float predicted_dx = vx * delay_time;
        float predicted_dy = vy * delay_time;

        // 计算预测的角度变化（基于位置变化）
        float angle_change = atan2(predicted_dy, predicted_dx) * 180.0f / CV_PI;

        // 考虑旋转状态
        float predicted_pitch = current_pitch;
        float predicted_yaw = current_yaw;

        if (is_spinning && abs(angular_velocity) > 0.1f) {
            // 旋转目标：基于角速度预测
            predicted_yaw += angular_velocity * delay_time * prediction_confidence;
        } else {
            // 非旋转目标：基于位置变化预测
            predicted_yaw += angle_change * 0.5f * prediction_confidence;
        }

        return Point3f(predicted_pitch, predicted_yaw, 0.0f);
    }

    Point2f predict() {
        state = kf.predict();

        // 激进预测：不限制移动距离，允许更远的前向预测
        if (has_last_pos) {
            float dx = state.at<float>(0) - last_valid_pos.x;
            float dy = state.at<float>(1) - last_valid_pos.y;
            float dist = sqrt(dx*dx + dy*dy);

            // 只做轻微限制，允许更大的预测距离
            float max_move = 100.0f;  // 增大到100像素，允许更激进预测
            if (dist > max_move) {
                float ratio = max_move / dist;
                state.at<float>(0) = last_valid_pos.x + dx * ratio;
                state.at<float>(1) = last_valid_pos.y + dy * ratio;
                state.at<float>(2) = dx * ratio;
                state.at<float>(3) = dy * ratio;
            }
            // 不限制时，保持原速度，让卡尔曼自由预测
            kf.statePost = state;
        }

        return Point2f(state.at<float>(0), state.at<float>(1));
    }

    Point2f update(const Point2f& detect_center) {
        last_valid_pos = detect_center;
        has_last_pos = true;

        measurement.at<float>(0) = detect_center.x;
        measurement.at<float>(1) = detect_center.y;

        Mat corrected_state = kf.correct(measurement);

        // 激进预测：不完全信任测量，保留更多速度分量
        // 混合预测和更新结果：位置更多信任更新，速度更多信任预测
        float position_gain = 0.3f;  // 位置增益
        float velocity_gain = 0.8f;  // 速度增益（更大，保持速度）

        state.at<float>(0) = corrected_state.at<float>(0) * position_gain + state.at<float>(0) * (1.0f - position_gain);
        state.at<float>(1) = corrected_state.at<float>(1) * position_gain + state.at<float>(1) * (1.0f - position_gain);
        state.at<float>(2) = corrected_state.at<float>(2) * velocity_gain + state.at<float>(2) * (1.0f - velocity_gain);
        state.at<float>(3) = corrected_state.at<float>(3) * velocity_gain + state.at<float>(3) * (1.0f - velocity_gain);

        kf.statePost = state;
        return Point2f(state.at<float>(0), state.at<float>(1));
    }

    void init(const Point2f& init_center) {
        last_valid_pos = init_center;
        has_last_pos = true;

        state.at<float>(0) = init_center.x;
        state.at<float>(1) = init_center.y;
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;
        kf.statePost = state;
    }

    void initWithVelocity(const Point2f& init_center, const Point2f& init_velocity) {
        last_valid_pos = init_center;
        has_last_pos = true;

        state.at<float>(0) = init_center.x;
        state.at<float>(1) = init_center.y;
        state.at<float>(2) = init_velocity.x;
        state.at<float>(3) = init_velocity.y;
        kf.statePost = state;
    }

    // ===================== 小陀螺检测 =====================
    bool detectSpinning(const Point2f& current_center) {
        // 优化：预先检查容量，避免频繁重新分配
        if (trajectory.size() >= TRAJECTORY_LENGTH) {
            trajectory.erase(trajectory.begin());
        }
        trajectory.push_back(current_center);

        if (trajectory.size() < TRAJECTORY_LENGTH / 2) {
            return is_spinning;
        }

        // 计算轨迹的中心
        Point2f center_sum(0, 0);
        for (const auto& p : trajectory) {
            center_sum += p;
        }
        Point2f trajectory_center = center_sum / (float)trajectory.size();

        // 计算平均角速度
        float total_angle = 0.0f;
        int angle_count = 0;
        for (size_t i = 1; i < trajectory.size(); i++) {
            float angle1 = atan2(trajectory[i-1].y - trajectory_center.y, trajectory[i-1].x - trajectory_center.x);
            float angle2 = atan2(trajectory[i].y - trajectory_center.y, trajectory[i].x - trajectory_center.x);
            float angle_diff = angle2 - angle1;
            if (angle_diff > CV_PI) angle_diff -= 2 * CV_PI;
            if (angle_diff < -CV_PI) angle_diff += 2 * CV_PI;
            total_angle += abs(angle_diff) * 180.0f / CV_PI;
            angle_count++;
        }

        angular_velocity = angle_count > 0 ? total_angle / angle_count : 0.0f;

        // 判断是否旋转
        if (angular_velocity > SPIN_ANGULAR_THRESHOLD) {
            spin_frame_count++;
            if (spin_frame_count >= SPIN_CONFIRM_FRAMES) {
                is_spinning = true;
            }
        } else {
            spin_frame_count--;
            if (spin_frame_count <= 0) {
                is_spinning = false;
                spin_frame_count = 0;
            }
        }

        return is_spinning;
    }

    // ===================== 自适应预测 =====================
    Point2f adaptivePredict(const Point2f& current_center) {
        if (!is_spinning || trajectory.size() < 10) {
            use_adaptive_predict = false;
            return predict();
        }

        use_adaptive_predict = true;

        // 计算旋转参数
        Point2f center_sum(0, 0);
        for (const auto& p : trajectory) {
            center_sum += p;
        }
        Point2f trajectory_center = center_sum / (float)trajectory.size();

        float avg_radius = 0.0f;
        float avg_angle = 0.0f;
        int valid_count = 0;

        for (const auto& p : trajectory) {
            float dx = p.x - trajectory_center.x;
            float dy = p.y - trajectory_center.y;
            float radius = sqrt(dx*dx + dy*dy);
            float angle = atan2(dy, dx);

            avg_radius += radius;
            avg_angle += angle;
            valid_count++;
        }

        if (valid_count < 2) return predict();

        avg_radius /= valid_count;
        avg_angle /= valid_count;

        // 预测下一帧角度（基于角速度）
        float predicted_angle = avg_angle + angular_velocity * dt * CV_PI / 180.0f;
        Point2f predicted_pos = trajectory_center + Point2f(
            cos(predicted_angle) * avg_radius,
            sin(predicted_angle) * avg_radius
        );

        // 计算预测置信度
        float prediction_error = 0.0f;
        for (const auto& p : trajectory) {
            float dx = p.x - predicted_pos.x;
            float dy = p.y - predicted_pos.y;
            prediction_error += sqrt(dx*dx + dy*dy);
        }
        prediction_error /= trajectory.size();
        prediction_confidence = max(0.3f, min(1.0f, 1.0f - prediction_error / 100.0f));

        // 结合卡尔曼预测和旋转预测
        Point2f kalman_pred = predict();
        Point2f adaptive_pred = predicted_pos;

        Point2f combined_pred = Point2f(
            kalman_pred.x * (1.0f - adapt_factor) + adaptive_pred.x * adapt_factor,
            kalman_pred.y * (1.0f - adapt_factor) + adaptive_pred.y * adapt_factor
        );

        state.at<float>(0) = combined_pred.x;
        state.at<float>(1) = combined_pred.y;
        kf.statePost = state;

        return combined_pred;
    }

    void reset() {
        has_last_pos = false;
        state = Mat::zeros(4, 1, CV_32F);
        setIdentity(kf.errorCovPost, Scalar::all(1));
        trajectory.clear();
        is_spinning = false;
        spin_frame_count = 0;
        angular_velocity = 0.0f;
        use_adaptive_predict = false;
        prediction_confidence = 1.0f;
    }
};

// ===================== 滑块回调 =====================
void onTrackbar(int, void*) {}

// ===================== 性能优化建议 =====================
// 1. 编译时启用优化：g++ -O3 -march=native rm3.cpp -o rm3 ...
// 2. 如需更高帧率，考虑使用多线程分离图像采集和处理
// 3. 使用海康相机的硬件触发模式可以进一步提高稳定性

// ===================== 串口发送结构 =====================
#pragma pack(1)
struct VisionSendData
{
    uint8_t header;
    float yaw;
    float pitch;
    float distance;
    uint8_t shoot;
    uint8_t checksum;
};
#pragma pack()

uint8_t checkSum(uint8_t* data, int len)
{
    uint8_t sum = 0;
    for(int i=0;i<len;i++)
        sum += data[i];
    return sum;
}

// ===================== 主函数 =====================
int main()
{
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
    MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[nIndex];

    // 创建句柄
    void* handle = NULL;
    nRet = MV_CC_CreateHandle(&handle, pDeviceInfo);
    if (nRet != MV_OK) {
        cout << "创建相机句柄失败！" << endl;
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

    // 设置曝光时间（90fps 需要 ≤ 11111μs，设置为 2000μs 以获得更好效果）
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
    int fd = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY | O_NDELAY);//串口路径
    if (fd == -1) {
        perror("串口打开失败");
    } else {
        struct termios options;
        tcgetattr(fd, &options);
        cfsetispeed(&options, B115200);
        cfsetospeed(&options, B115200);
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;
        options.c_cflag &= ~PARENB;
        options.c_cflag &= ~CSTOPB;
        options.c_cflag &= ~CRTSCTS;
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        options.c_oflag &= ~OPOST;
        tcsetattr(fd, TCSANOW, &options);
    }

    // ===================== 图像处理循环 =====================
    Mat frame, hsv, binary, Gaussian, dilatee, morph;
    Mat element_small = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat element_medium = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat element_large = getStructuringElement(MORPH_RECT, Size(9, 9));
    Mat element_huge = getStructuringElement(MORPH_RECT, Size(15, 15));
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // 优化：预分配轮廓和层级容器容量，减少动态分配
    contours.reserve(500);
    hierarchy.reserve(500);

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
        nRet = MV_CC_GetOneFrameTimeout(handle, pData, nDataSize, &stImageInfo, 10);
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

        // 帧率统计
        static int frame_counter = 0;
        static auto fps_start_time = chrono::steady_clock::now();
        frame_counter++;
        auto current_time = chrono::steady_clock::now();
        auto fps_elapsed = chrono::duration_cast<chrono::milliseconds>(current_time - fps_start_time).count();
        if (fps_elapsed >= 1000) {
            float fps = frame_counter * 1000.0f / fps_elapsed;
            cout << "[FPS] 当前帧率: " << fixed << setprecision(1) << fps << " fps" << endl;
            frame_counter = 0;
            fps_start_time = current_time;
        }

        // 转换为 OpenCV Mat（RGB格式）
        Mat rgb(stImageInfo.nHeight, stImageInfo.nWidth, CV_8UC3, pData);
        Mat frame;
        cvtColor(rgb, frame, COLOR_RGB2BGR);  // 转换为BGR格式供OpenCV显示

        if (frame.empty()) continue;

        // 优化：降低分辨率进行图像处理，提高帧率
        Mat frame_small;
        resize(frame, frame_small, Size(), 0.5, 0.5, INTER_LINEAR);  // 缩小到50%

        Mat frame_copy = frame_small.clone();  // 使用缩小后的图像进行处理
        bool is_armor_detected = false;
        Point2f detect_center;

        // 转灰度图
        Mat gray;
        cvtColor(frame_small, gray, COLOR_BGR2GRAY);

        // 降低阈值以识别红色灯条（80-255）
        threshold(gray, binary, 120, 255, THRESH_BINARY);

        // 优化：使用更小的高斯核以减少计算量
        GaussianBlur(binary, binary, Size(3, 3), 0.5);

        // 形态学操作：开运算去噪 + 闭运算填充
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));  // 减小核大小
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);

        // 膨胀连接断裂区域（减少膨胀次数）
        dilate(binary, dilatee, element_small, Point(-1, -1), 1);  // 减少膨胀次数
        dilate(dilatee, dilatee, element_small, Point(-1, -1), 1);

        // 优化：清空轮廓容器，而非重新分配
        contours.clear();
        hierarchy.clear();

        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        // 优化：预分配lightInfos容量
        vector<LightDescriptor> lightInfos;
        lightInfos.reserve(contours.size() / 4);  // 估计约1/4的轮廓是有效灯条

        for (int i = 0; i < contours.size(); i++) {
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

        // 优化：定期清理未使用的内存（每30帧一次）
        static int memory_cleanup_count = 0;
        if (++memory_cleanup_count >= 30) {
            memory_cleanup_count = 0;
            contours.shrink_to_fit();
            hierarchy.shrink_to_fit();
            lightInfos.shrink_to_fit();
        }

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
                float y = tvec.at<double>(1) / 1000.0f;
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
                if (fd != -1 && distance_real > 0.5f && distance_real < 10.0f) {
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
                        send_data.checksum = checkSum((uint8_t*)&send_data, sizeof(send_data)-1);
                        int n = write(fd, &send_data, sizeof(send_data));
                        if (n < 0) perror("串口写入失败");

                        last_send_center = detect_center;  // 更新上次发送位置
                    } else {
                        // 不稳定帧，跳过发送或发送上一次稳定的数据
                        static int skip_frame_count = 0;
                        if (++skip_frame_count % 50 == 0) {  // 减少输出频率
                            cout << "【跳帧】检测位置不稳定，跳过发送（跳变距离: "
                                 << sqrt(pow(detect_center.x - last_send_center.x, 2) +
                                        pow(detect_center.y - last_send_center.y, 2)) << "）" << endl;
                        }
                    }
                }

                // 每帧输出检测信息
                cout << "检测中心: (" << detect_center.x << ", " << detect_center.y << ")" << endl;
                cout << "距离: " << fixed << setprecision(2) << distance_m << " m" << endl;
                cout << "Yaw: " << fixed << setprecision(2) << yaw << " 度" << endl;
                cout << "预测置信度: " << fixed << setprecision(2) << armor_kalman.prediction_confidence << endl;

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
    contours.clear();
    contours.shrink_to_fit();
    hierarchy.clear();
    hierarchy.shrink_to_fit();

    if (fd != -1) close(fd);
    cv::destroyAllWindows();
    return 0;
}