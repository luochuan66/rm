#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

// 先包含标准库
#include <vector>
#include <string>

// 再包含OpenCV
#include <opencv2/opencv.hpp>

class ArmorKalmanFilter {
private:
    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat measurement;
    cv::Point2f last_valid_pos;
    bool has_last_pos;

private:
    // 小陀螺检测相关
    std::vector<cv::Point2f> trajectory;
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

    ArmorKalmanFilter(float dt_ = 0.2f);
    
    // ===================== 延迟补偿预测 =====================
    cv::Point2f predictWithDelayCompensation(float delay_time);
    
    // ===================== 带延迟的姿态预测 =====================
    cv::Point3f predictPoseWithDelay(float delay_time, float current_pitch, float current_yaw);
    
    cv::Point2f predict();
    cv::Point2f update(const cv::Point2f& detect_center);
    void init(const cv::Point2f& init_center);
    void initWithVelocity(const cv::Point2f& init_center, const cv::Point2f& init_velocity);
    
    // ===================== 小陀螺检测 =====================
    bool detectSpinning(const cv::Point2f& current_center);
    
    // ===================== 自适应预测 =====================
    cv::Point2f adaptivePredict(const cv::Point2f& current_center);
    void reset();
};

#endif // KALMAN_FILTER_H
