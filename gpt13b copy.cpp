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
const float IMAGE_DELAY = 0.033f;       // 图像采集延迟 (30fps ≈ 33ms)
const float PROCESS_DELAY = 0.010f;      // 图像处理延迟 (10ms)
const float TRANSMISSION_DELAY = 0.005f; // 串口传输延迟 (5ms)
const float MECHANICAL_DELAY = 0.050f;  // 机械响应延迟 (50ms)
const float TOTAL_DELAY = IMAGE_DELAY + PROCESS_DELAY + TRANSMISSION_DELAY + MECHANICAL_DELAY; // 总延迟约98ms

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

    // 弹道在线标定
    vector<float> pitch_errors;
    vector<float> distance_samples;
    vector<Point3f> history_positions;  // 历史位置(x, y, z)
    const int MAX_ERROR_SAMPLES = 100;
    const int MAX_HISTORY_POSITIONS = 50;
    float bullet_speed_variance;
    const float DEFAULT_BULLET_SPEED = 28.0f;
    int sample_count;

    // 自适应预测参数
    float adapt_factor;
    float dt;

public:
    // 公开成员变量（用于显示和调试）
    float angular_velocity;
    bool is_spinning;
    bool use_adaptive_predict;
    float prediction_confidence;
    float bullet_speed_estimate;

    ArmorKalmanFilter(float dt_ = 0.2f) : dt(dt_), has_last_pos(false),
                                          angular_velocity(0.0f), is_spinning(false), spin_frame_count(0),
                                          bullet_speed_estimate(DEFAULT_BULLET_SPEED),
                                          bullet_speed_variance(1.0f), sample_count(0),
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

    float ballisticCompensation(float distance, float height, float bullet_speed) {
        const float g = 9.8f;
        float t = distance / bullet_speed;
        float drop = 0.5f * g * t * t;
        float target_y = height + drop;
        return atan2(target_y, distance) * 180.0f / CV_PI;
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
        trajectory.push_back(current_center);
        if (trajectory.size() > TRAJECTORY_LENGTH) {
            trajectory.erase(trajectory.begin());
        }

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

    // ===================== 弹道在线标定 =====================
    void updateBallisticCalibration(float actual_distance, float measured_pitch, float measured_yaw,
                                    const Point3f& position_3d) {
        history_positions.push_back(position_3d);
        if (history_positions.size() > MAX_HISTORY_POSITIONS) {
            history_positions.erase(history_positions.begin());
        }

        // 计算理论弹道补偿
        float theoretical_pitch = ballisticCompensation(actual_distance, position_3d.y, bullet_speed_estimate);
        float pitch_error = theoretical_pitch - measured_pitch;

        if (abs(pitch_error) < 5.0f) {  // 只在误差合理时更新
            pitch_errors.push_back(pitch_error);
            distance_samples.push_back(actual_distance);

            if (pitch_errors.size() > MAX_ERROR_SAMPLES) {
                pitch_errors.erase(pitch_errors.begin());
                distance_samples.erase(distance_samples.begin());
            }

            // 更新弹道速度估计（基于误差反馈）
            sample_count++;
            float error_sum = 0.0f;
            for (float e : pitch_errors) {
                error_sum += e;
            }
            float avg_error = error_sum / pitch_errors.size();

            // 自适应调整弹道速度
            float speed_adjustment = avg_error * 0.1f;
            bullet_speed_estimate += speed_adjustment;
            bullet_speed_estimate = max(20.0f, min(35.0f, bullet_speed_estimate));  // 限制在合理范围

            // 更新方差（用于置信度计算）
            float variance_sum = 0.0f;
            for (float e : pitch_errors) {
                variance_sum += (e - avg_error) * (e - avg_error);
            }
            bullet_speed_variance = variance_sum / pitch_errors.size();
        }
    }

    float getAdaptivePitch(float distance, float height, float current_yaw) {
        // 根据历史数据自适应调整pitch
        float base_pitch = ballisticCompensation(distance, height, bullet_speed_estimate);

        // 添加距离相关的补偿
        float dist_compensation = 0.0f;
        if (distance_samples.size() >= 10) {
            float dist_sum = 0.0f;
            for (float d : distance_samples) {
                dist_sum += d;
            }
            float avg_distance = dist_sum / distance_samples.size();
            float dist_diff = distance - avg_distance;
            dist_compensation = dist_diff * 0.5f;  // 距离差异补偿
        }

        // 添加旋转状态补偿
        float spin_compensation = 0.0f;
        if (is_spinning) {
            spin_compensation = angular_velocity * 0.1f * prediction_confidence;
        }

        return base_pitch + dist_compensation + spin_compensation;
    }

    float getConfidence() const {
        if (pitch_errors.size() < 10) return 0.5f;
        return max(0.1f, min(1.0f, 1.0f - sqrt(bullet_speed_variance) / 3.0f));
    }

    void reset() {
        has_last_pos = false;
        state = Mat::zeros(4, 1, CV_32F);
        setIdentity(kf.errorCovPost, Scalar::all(1));
        trajectory.clear();
        is_spinning = false;
        spin_frame_count = 0;
        angular_velocity = 0.0f;
        pitch_errors.clear();
        distance_samples.clear();
        history_positions.clear();
        sample_count = 0;
        use_adaptive_predict = false;
        prediction_confidence = 1.0f;
        bullet_speed_estimate = DEFAULT_BULLET_SPEED;
    }
};

// ===================== 滑块回调 =====================
void onTrackbar(int, void*) {}

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

    VideoCapture video(2);
    if (!video.isOpened()) {
        cout << "摄像头打开失败！请检查外接摄像头是否连接" << endl;
        return -1;
    }

    // 设置相机参数
    video.set(CAP_PROP_BUFFERSIZE, 20);  // 设置缓冲区为20帧
    video.set(CAP_PROP_FPS, 30);         // 设置帧率

    // 降低曝光以减少白色背景和噪点
    video.set(CAP_PROP_AUTO_EXPOSURE, 0.25);  // 关闭自动曝光
    video.set(CAP_PROP_EXPOSURE, 3);         // 设置曝光时间（数值越小曝光越低）

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

    for (;;) {
        // 释放相机缓存，只取最新帧
        for (int i = 0; i < 2; i++) {
            Mat temp;
            video >> temp;
        }

        video >> frame;
        if (frame.empty()) break;
        Mat frame_copy = frame.clone();
        bool is_armor_detected = false;
        Point2f detect_center;

        // 转灰度图
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 固定阈值二值化（160-255）
        Mat binary;
        threshold(gray, binary, 160, 255, THRESH_BINARY);

        // 输出阈值信息（每30帧输出一次）
        static int frame_count = 0;
        if (++frame_count % 30 == 0) {
            cout << "[预处理] 使用固定阈值二值化: 160-255" << endl;
        }

        // 高斯模糊
        GaussianBlur(binary, binary, Size(5, 5), 1.0);

        // 形态学操作：开运算去噪 + 闭运算填充
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);

        // 膨胀连接断裂区域
        dilate(binary, dilatee, element_small, Point(-1, -1), 2);
        dilate(dilatee, dilatee, element_medium, Point(-1, -1), 1);

        findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        vector<LightDescriptor> lightInfos;
        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 80 || contours[i].size() <= 10) continue;

            RotatedRect Light_Rec = fitEllipse(contours[i]);
            if (Light_Rec.size.width / Light_Rec.size.height > 5) continue;
            if (Light_Rec.size.area() < 60) continue;

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

                if (estimated_distance > 1800.0f) {
                    // 远距离(>1.8m)
                    max_spacing_error = use_tracking_bonus ? 0.50 : 0.35;  // 有跟踪时放宽到50%
                } else if (estimated_distance > 1200.0f) {
                    // 中距离(1.2-1.8m)
                    max_spacing_error = use_tracking_bonus ? 0.60 : 0.45;  // 有跟踪时放宽到60%
                } else {
                    // 近距离(≤1.2m)
                    max_spacing_error = use_tracking_bonus ? 0.70 : 0.55;  // 有跟踪时放宽到70%
                }

                // 关键过滤：灯条间距必须符合物理约束
                if (spacing_error_ratio > max_spacing_error) continue;

                if (angleGap_ > 20 || LenGap_ratio > 1.2 || lengap_ratio > 1.0 ||
                    yGap_ratio > 2.0 || xGap_ratio > 3.0 || xGap_ratio < 0.6 ||
                    ratio > 4 || ratio < 0.6) continue;

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

                // 使用自适应弹道补偿
                Point3f position_3d(x, y, z);
                float pitch_comp = armor_kalman.getAdaptivePitch(distance_real, y, yaw);

                // 更新弹道在线标定
                armor_kalman.updateBallisticCalibration(distance_real, pitch_comp, yaw, position_3d);

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
                        send_data.pitch = pitch_comp + delay_compensated_pose.x * 0.3f;  // 考虑延迟补偿的pitch调整
                        send_data.distance = distance_real;
                        send_data.shoot = (lost_frame_count == 0 && armor_kalman.prediction_confidence > 0.6f) ? 1 : 0;  // 高置信度才射击
                        send_data.checksum = checkSum((uint8_t*)&send_data, sizeof(send_data)-1);
                        int n = write(fd, &send_data, sizeof(send_data));
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
                cout << "弹道速度: " << fixed << setprecision(2) << armor_kalman.bullet_speed_estimate << " m/s" << endl;
                cout << "标定置信度: " << fixed << setprecision(2) << armor_kalman.getConfidence() << endl;
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

        
        if (is_kalman_init && kalman_center.x > 0 && kalman_center.y > 0) {
            circle(frame_copy, kalman_center, 4, Scalar(0, 0, 255), -1);

            string predText = armor_kalman.use_adaptive_predict ? "Adaptive" : "Kalman";
            putText(frame_copy, predText, kalman_center + Point2f(10, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

            // 小陀螺状态显示
            if (armor_kalman.is_spinning) {
                string spinText = "SPIN: " + to_string(static_cast<int>(armor_kalman.angular_velocity));
                putText(frame_copy, spinText, kalman_center + Point2f(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            }

            // 标定信息显示
            string calibText = "V: " + to_string(static_cast<int>(armor_kalman.bullet_speed_estimate));
            putText(frame_copy, calibText, kalman_center + Point2f(10, 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        imshow("binary", binary);
        imshow("video", frame_copy);
        if (waitKey(50) == 27) break;
    }

    video.release();
    if (fd != -1) close(fd);
    cv::destroyAllWindows();
    return 0;
}