#ifndef ARMOR_DETECTOR_H
#define ARMOR_DETECTOR_H

// 先包含标准库
#include <vector>
#include <string>

// 再包含OpenCV
#include <opencv2/opencv.hpp>

// ===================== 装甲板物理尺寸（单位：mm） =====================
const float ARMOR_WIDTH = 140.0f;    
const float ARMOR_HEIGHT = 125.0f;   

// ===================== 相机内参（标定结果） =====================
const cv::Mat CAMERA_MATRIX = (cv::Mat_<double>(3, 3) <<
    589.0427686678858, 0.0, 1041.580514579736,
    0.0, 589.3795612894731, 534.8622665285162,
    0.0, 0.0, 1.0);
const cv::Mat DIST_COEFFS = (cv::Mat_<double>(5, 1) <<
    0.002445382552691366, -0.01440570673944744, -0.0001341828334394887,
    -0.0002228839280500997, -0.0006838812197758657);

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
bool solveArmorPnP(const std::vector<cv::Point2f>& vertices, cv::Mat& rvec, cv::Mat& tvec);
cv::Vec3f rvecToEuler(const cv::Mat& rvec);

#endif // ARMOR_DETECTOR_H
