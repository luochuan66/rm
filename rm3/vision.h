#ifndef RM3_VISION_H
#define RM3_VISION_H

#include <opencv2/opencv.hpp>
#include "algorithm.h"

using namespace std;
using namespace cv;

// ===================== 图像处理类 =====================
class VisionProcessor {
private:
    Mat element_small, element_medium, element_large, element_huge;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

public:
    Point2f detect_center;
    float distance_m;
    float pitch, yaw;
    float prediction_confidence;
    bool is_detected;

public:
    VisionProcessor() : is_detected(false) {
        element_small = getStructuringElement(MORPH_RECT, Size(3, 3));
        element_medium = getStructuringElement(MORPH_RECT, Size(5, 5));
        element_large = getStructuringElement(MORPH_RECT, Size(9, 9));
        element_huge = getStructuringElement(MORPH_RECT, Size(15, 15));

        // 预分配轮廓和层级容器容量，减少动态分配
        contours.reserve(500);
        hierarchy.reserve(500);
    }

    // ===================== 图像预处理 =====================
    Mat preprocess(const Mat& frame) {
        Mat gray, binary;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 降低阈值以识别红色灯条（80-255）
        threshold(gray, binary, 120, 255, THRESH_BINARY);

        // 优化：使用更小的高斯核以减少计算量
        GaussianBlur(binary, binary, Size(3, 3), 0.5);

        // 形态学操作：开运算去噪 + 闭运算填充
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));  // 减小核大小
        morphologyEx(binary, binary, MORPH_OPEN, kernel);
        morphologyEx(binary, binary, MORPH_CLOSE, kernel);

        // 膨胀连接断裂区域（减少膨胀次数）
        Mat dilatee;
        dilate(binary, dilatee, element_small, Point(-1, -1), 1);  // 减少膨胀次数
        dilate(dilatee, dilatee, element_small, Point(-1, -1), 1);

        return dilatee;
    }

    // ===================== 提取灯条 =====================
    vector<LightDescriptor> extractLights(const Mat& binary) {
        vector<LightDescriptor> lightInfos;

        // 优化：清空轮廓容器，而非重新分配
        contours.clear();
        hierarchy.clear();

        findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        // 优化：预分配lightInfos容量
        lightInfos.reserve(contours.size() / 4);  // 估计约1/4的轮廓是有效灯条

        for (int i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 80 || contours[i].size() <= 10) continue;

            RotatedRect Light_Rec = fitEllipse(contours[i]);

            // 计算长宽比（长边/短边）
            float aspect_ratio = max(Light_Rec.size.width, Light_Rec.size.height) /
                                 min(Light_Rec.size.width, Light_Rec.size.height);
            // 针对中距离优化：长宽比 < 1.8 或 > 8 的轮廓
            if (aspect_ratio < 2.5 || aspect_ratio > 6.5) continue;

            // 针对中距离优化：面积过滤更宽松
            if (Light_Rec.size.area() < 50) continue;

            lightInfos.push_back(LightDescriptor(Light_Rec));
        }

        return lightInfos;
    }

    // ===================== 匹配装甲板 =====================
    bool matchArmor(const vector<LightDescriptor>& lightInfos,
                   Point2f& armorCenter,
                   vector<Point2f>& vertices,
                   float& estimated_distance,
                   const Point2f& last_armor_center,
                   const Point2f& kalman_center,
                   bool is_kalman_init,
                   float last_valid_spacing,
                   float last_valid_distance,
                   int valid_detection_count) {

        const int MIN_VALID_FRAMES_FOR_TRACKING = 5;

        vector<bool> lightUsed(lightInfos.size(), false);
        bool armorFound = false;
        Point2f bestArmorCenter;
        float bestArmorScore = -1;
        int bestLightI = -1, bestLightJ = -1;

        for (size_t i = 0; i < lightInfos.size(); i++) {
            for (size_t j = i + 1; j < lightInfos.size(); j++) {
                const LightDescriptor& leftLight = lightInfos[i];
                const LightDescriptor& rightLight = lightInfos[j];
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

                float estimated_dist = (LIGHT_BAR_PHYSICAL_LENGTH * 500.0f) / meanLen;  // 估计距离(mm)
                float theoretical_spacing_px = (LIGHT_BAR_SPACING * 500.0f) / estimated_dist;  // 理论间距(像素)
                float spacing_error_ratio = abs(dis - theoretical_spacing_px) / theoretical_spacing_px;  // 间距误差比

                // 根据连续性跟踪调整间距误差容忍度
                float max_spacing_error;
                bool use_tracking_bonus = (valid_detection_count >= MIN_VALID_FRAMES_FOR_TRACKING) &&
                                         (last_valid_spacing > 0);

                // 针对主要识别距离 2.2m 优化
                if (estimated_dist > 2500.0f) {
                    // 远距离(>2.5m)
                    max_spacing_error = use_tracking_bonus ? 0.80 : 0.65;
                } else if (estimated_dist > 1800.0f) {
                    // 中远距离(1.8-2.5m) - 主要识别范围
                    max_spacing_error = use_tracking_bonus ? 0.90 : 0.75;
                } else if (estimated_dist > 1200.0f) {
                    // 中距离(1.2-1.8m)
                    max_spacing_error = use_tracking_bonus ? 1.00 : 0.85;
                } else {
                    // 近距离(≤1.2m)
                    max_spacing_error = use_tracking_bonus ? 1.10 : 0.95;
                }

                // 关键过滤：灯条间距必须符合物理约束
                if (spacing_error_ratio > max_spacing_error) continue;

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
                if (angleGap_ > 30 || LenGap_ratio > 2.0 || lengap_ratio > 1.5 ||
                    yGap_ratio > 3.0 || xGap_ratio > 4.0 || xGap_ratio < 0.5 ||
                    ratio > 3.5 || ratio < 1.5) continue;

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
            armorCenter = bestArmorCenter;

            const LightDescriptor& leftLight = lightInfos[bestLightI];
            const LightDescriptor& rightLight = lightInfos[bestLightJ];
            float dis = sqrt(pow(leftLight.center.x - rightLight.center.x, 2) + pow(leftLight.center.y - rightLight.center.y, 2));
            float meanLen = (leftLight.length + rightLight.length) / 2;

            // 更新连续性跟踪参数
            const float LIGHT_BAR_PHYSICAL_LENGTH = 80.0f;
            estimated_distance = (LIGHT_BAR_PHYSICAL_LENGTH * 500.0f) / meanLen;

            RotatedRect rect = RotatedRect(armorCenter, Size(dis, meanLen), (leftLight.angle + rightLight.angle) / 2);
            Point2f rect_vertices[4];
            rect.points(rect_vertices);
            vertices.assign(rect_vertices, rect_vertices + 4);
        }

        return armorFound;
    }

    // ===================== 计算姿态 =====================
    bool computePose(const vector<Point2f>& vertices, Point2f& detected_center) {
        Mat rvec, tvec;
        if (!solveArmorPnP(vertices, rvec, tvec)) {
            return false;
        }

        float distance = (float)tvec.at<double>(2);
        distance_m = distance / 1000.0f;
        Vec3f euler = rvecToEuler(rvec);
        pitch = euler[0];
        yaw = euler[1];

        float x = tvec.at<double>(0) / 1000.0f;
        float y = tvec.at<double>(1) / 1000.0f;
        float z = tvec.at<double>(2) / 1000.0f;

        return true;
    }

    // ===================== 绘制结果 =====================
    void drawResult(Mat& frame, const Point2f& detect_center, const vector<Point2f>& vertices,
                   const Point2f& kalman_center, const ArmorKalmanFilter& kalman) {
        // 绘制装甲板框
        for (int k = 0; k < 4; k++) {
            line(frame, vertices[k], vertices[(k + 1) % 4], Scalar(0, 0, 255), 2.2);
        }

        // 绘制检测中心
        circle(frame, detect_center, 4, Scalar(255, 0, 0), -1);
        putText(frame, "Detect", detect_center + Point2f(10, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);

        // 绘制距离信息
        string distText = "Dist: " + to_string(static_cast<int>(distance_m * 10) / 10.0f) + "m";
        putText(frame, distText, detect_center + Point2f(10, 25), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

        // 绘制卡尔曼预测中心
        if (kalman_center.x > 0 && kalman_center.y > 0) {
            circle(frame, kalman_center, 4, Scalar(0, 0, 255), -1);

            string predText = kalman.use_adaptive_predict ? "Adaptive" : "Kalman";
            putText(frame, predText, kalman_center + Point2f(10, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);

            // 小陀螺状态显示
            if (kalman.is_spinning) {
                string spinText = "SPIN: " + to_string(static_cast<int>(kalman.angular_velocity));
                putText(frame, spinText, kalman_center + Point2f(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            }
        }
    }

    // ===================== 内存清理 =====================
    void cleanup() {
        contours.clear();
        contours.shrink_to_fit();
        hierarchy.clear();
        hierarchy.shrink_to_fit();
    }
};

#endif // RM3_VISION_H
