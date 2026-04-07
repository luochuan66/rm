#ifndef RM3_ALGORITHM_H
#define RM3_ALGORITHM_H

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <cmath>

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

// ===================== 灯条类 =====================
class LightDescriptor
{
public:
    float width, length, angle, area;
    cv::Point2f center;
public:
    LightDescriptor() {};
    LightDescriptor(const cv::RotatedRect& light)//构造函数提取和保存灯条的信息
    {
        width = light.size.width;//宽
        length = light.size.height;//长
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};

// ===================== PnP位姿解算 =====================
bool solveArmorPnP(const vector<Point2f>& vertices, Mat& rvec, Mat& tvec) {//四个角点的图像坐标，输出旋转向量和平移向量，2d-3d
    vector<Point3f> objectPoints = {//定义
        Point3f(-ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0),//左上
        Point3f(ARMOR_WIDTH/2, -ARMOR_HEIGHT/2, 0),//右上
        Point3f(ARMOR_WIDTH/2, ARMOR_HEIGHT/2, 0),
        Point3f(-ARMOR_WIDTH/2, ARMOR_HEIGHT/2, 0)
    };
    return solvePnP(objectPoints, vertices, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, false, SOLVEPNP_ITERATIVE);//pnp
}//输入3D点、2D点、相机内参、畸变系数 输出旋转向量和平移向量

Vec3f rvecToEuler(const Mat& rvec) {////旋转向量转欧拉角
    Mat R;
    Rodrigues(rvec, R);//旋转向量转旋转矩阵
    float yaw = atan2(R.at<double>(1, 0), R.at<double>(0, 0)) * 180.0f / CV_PI;//不理解但运用
    float pitch = atan2(-R.at<double>(2, 0), sqrt(R.at<double>(2, 1) * R.at<double>(2, 1) +
                                                      R.at<double>(2, 2) * R.at<double>(2, 2))) * 180.0f / CV_PI;
    float roll = atan2(R.at<double>(2, 1), R.at<double>(2, 2)) * 180.0f / CV_PI;
    return Vec3f(pitch, yaw, roll);
}

// ===================== 卡尔曼滤波与小陀螺检测 =====================
class ArmorKalmanFilter {
private:
    KalmanFilter kf;//对象
    Mat state;//状态向量 [x, y, vx, vy]
    Mat measurement;//测量向量 x y
    Point2f last_valid_pos;//上次位置
    bool has_last_pos;//有五历史记录

private:
    // 小陀螺检测相关
    vector<Point2f> trajectory;//历史运动轨迹点
    const int TRAJECTORY_LENGTH = 30;//轨迹队列的最大长度
    int spin_frame_count;//旋转状态计数器
    const int SPIN_CONFIRM_FRAMES = 15;//确认旋转所需的最小连续帧数
    const float SPIN_ANGULAR_THRESHOLD = 30.0f;  // 角速度阈值(度/帧)

    float adapt_factor;//自适应预测因子？
    float dt;//时间

public:
    // 公开成员变量（用于显示和调试）
    float angular_velocity;//角速度
    bool is_spinning;//是否旋转
    bool use_adaptive_predict;//是否启用自适应预测
    float prediction_confidence;//预测置信度因子

    ArmorKalmanFilter(float dt_ = 0.022f) : dt(dt_), has_last_pos(false),//初始化，dt为时间间隔
                                          angular_velocity(0.0f), is_spinning(false), spin_frame_count(0),
                                          use_adaptive_predict(false), adapt_factor(0.5f), prediction_confidence(1.0f) {
        kf = KalmanFilter(4, 2, 0);//初始化
        kf.transitionMatrix = (Mat_<float>(4, 4) <<
            1, 0, dt, 0,//直线运动公式
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1);
        kf.measurementMatrix = (Mat_<float>(2, 4) <<
            1, 0, 0, 0,//x   
            0, 1, 0, 0);//y
        // 激进预测：增大过程噪声，允许更大的速度变化
        kf.processNoiseCov = (Mat_<float>(4, 4) <<////过程噪声协方差
            0.5, 0, 0, 0,  // x位置噪声
            0, 0.5, 0, 0,  // y位置噪声
            0, 0, 5.0, 0,  // vx速度噪声
            0, 0, 0, 5.0); // vy速度噪声
        // 减小测量噪声，更多相信预测而非测量
        kf.measurementNoiseCov = (Mat_<float>(2, 2) <<//测量噪声协方差
            0.05, 0,
            0, 0.05);//越小越相信预测，越大越相信测量
        setIdentity(kf.errorCovPost, Scalar::all(1));//设置误差协方差矩阵
        state = Mat::zeros(4, 1, CV_32F);//初始化状态向量
        measurement = Mat::zeros(2, 1, CV_32F);//测量向量
    }

    // ===================== 延迟补偿预测 =====================
    Point2f predictWithDelayCompensation(float delay_time) {//预测步数 = 延迟时间 / 每帧时间
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
            float dy = state.at<float>(1) - last_valid_pos.y;//算差
            float dist = sqrt(dx*dx + dy*dy);//计算实际移动距离
            float ratio = 1.0f;//初始化比例因子

            if (dist > max_move) {//距离限制逻辑
                ratio = max_move / dist;//缩小因子
                state.at<float>(0) = last_valid_pos.x + dx * ratio;//修正预测位置
                state.at<float>(1) = last_valid_pos.y + dy * ratio;
            }

            kf.statePost = state;//更新状态向量
        }

        return Point2f(state.at<float>(0), state.at<float>(1));//返回点
    }

    // ===================== 带延迟的姿态预测 =====================
    Point3f predictPoseWithDelay(float delay_time, float current_pitch, float current_yaw) {
        // 预测延迟后的pitch和yaw
        int prediction_steps = static_cast<int>(delay_time / dt);//预测步数计算
        prediction_steps = max(1, min(prediction_steps, 10));//10步

        // 获取速度
        float vx = state.at<float>(2);
        float vy = state.at<float>(3);

        // 预测位置变化
        float predicted_dx = vx * delay_time;
        float predicted_dy = vy * delay_time;

        // 计算预测的角度变化（基于位置变化）
        float angle_change = atan2(predicted_dy, predicted_dx) * 180.0f / CV_PI;

        // 考虑旋转状态
        float predicted_pitch = current_pitch;//初始化
        float predicted_yaw = current_yaw;

        if (is_spinning && abs(angular_velocity) > 0.1f) {//判断是不是小脱罗
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
            float dx = state.at<float>(0) - last_valid_pos.x;//计算预测位置与上一帧实际位置的差值
            float dy = state.at<float>(1) - last_valid_pos.y;
            float dist = sqrt(dx*dx + dy*dy);//欧拉距离

            // 只做轻微限制，允许更大的预测距离
            float max_move = 100.0f;  // 增大到100像素，允许更激进预测
            if (dist > max_move) {
                float ratio = max_move / dist;//计算缩放比例
                state.at<float>(0) = last_valid_pos.x + dx * ratio;//修正位置更新
                state.at<float>(1) = last_valid_pos.y + dy * ratio;
                state.at<float>(2) = dx * ratio;
                state.at<float>(3) = dy * ratio;//数独更新
            }
            // 不限制时，保持原速度，让卡尔曼自由预测
            kf.statePost = state;//更新
        }

        return Point2f(state.at<float>(0), state.at<float>(1));
    }

   // ===================== 卡尔曼更新（带状态混合策略）=====================
Point2f update(const Point2f& detect_center) {
    // 这个位置将用于下一次预测时的距离限制
    last_valid_pos = detect_center;
    has_last_pos = true;

    // 测量值就是算法检测到的目标中心位置
    measurement.at<float>(0) = detect_center.x;
    measurement.at<float>(1) = detect_center.y;

    // 执行卡尔曼修正
    // corrected_state 包含卡尔曼根据测量值计算出的"最优估计"
    // 包含: [x_corrected, y_corrected, vx_corrected, vy_corrected]
    Mat corrected_state = kf.correct(measurement);

    // ===== 步骤4：激进预测策略 - 状态混合 =====
    // 设计思想：不完全信任测量值，保留更多速度信息
    // 这样可以让滤波器更"激进"地预测高速运动目标
    float position_gain = 0.3f;   // 位置增益（较小 → 更信任预测值）
    float velocity_gain = 0.8f;   // 速度增益（较大 → 保持速度惯性）

    // 4.1 位置混合：30% 修正值 + 70% 预测值
    // 目的：平滑位置变化，减少检测抖动的影响
    // 例如：检测位置跳变5像素，输出只跳变1.5像素
    state.at<float>(0) = corrected_state.at<float>(0) * position_gain + 
                         state.at<float>(0) * (1.0f - position_gain);
    state.at<float>(1) = corrected_state.at<float>(1) * position_gain + 
                         state.at<float>(1) * (1.0f - position_gain);
    
    // 4.2 速度混合：80% 预测值 + 20% 修正值
    // 目的：保持速度惯性，避免速度突变
    // 例如：目标突然停止，速度会缓慢下降而不是立即归零
    state.at<float>(2) = corrected_state.at<float>(2) * velocity_gain + 
                         state.at<float>(2) * (1.0f - velocity_gain);
    state.at<float>(3) = corrected_state.at<float>(3) * velocity_gain + 
                         state.at<float>(3) * (1.0f - velocity_gain);

    // 步骤5：将混合后的状态写回卡尔曼滤波器
    // 这样下一次 predict() 会基于这个混合状态进行预测
    kf.statePost = state;
    
    // 步骤6：返回平滑后的位置
    return Point2f(state.at<float>(0), state.at<float>(1));
}

    void init(const Point2f& init_center) {//无初速度
        last_valid_pos = init_center;//保存
        has_last_pos = true;//标记
        state.at<float>(0) = init_center.x;//x坐标
        state.at<float>(1) = init_center.y;//y坐标
        state.at<float>(2) = 0;//x速度
        state.at<float>(3) = 0;//y速度
        kf.statePost = state;
    }

    void initWithVelocity(const Point2f& init_center, const Point2f& init_velocity) {//有初速度
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
        if (trajectory.size() >= TRAJECTORY_LENGTH) {//轨迹管理
            trajectory.erase(trajectory.begin());//看有没有多的，之维护一个
        }
        trajectory.push_back(current_center);

        if (trajectory.size() < TRAJECTORY_LENGTH / 2) {// 最小数据量检查
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
            spin_frame_count--;//不转
            if (spin_frame_count <= 0) {
                is_spinning = false;
                spin_frame_count = 0;
            }
        }

        return is_spinning;
    }

    // ===================== 自适应预测 =====================//根据目标运动状态自动调整预测模型的智能跟踪方法
    Point2f adaptivePredict(const Point2f& current_center) {
        if (!is_spinning || trajectory.size() < 10) {
            use_adaptive_predict = false;//// 回退到普通预测（线性运动）
            return predict();
        }

        use_adaptive_predict = true;

        // 计算旋转参数
        Point2f center_sum(0, 0);//计算轨迹中心
        for (const auto& p : trajectory) {
            center_sum += p;
        }
        Point2f trajectory_center = center_sum / (float)trajectory.size();

        float avg_radius = 0.0f;
        float avg_angle = 0.0f;
        int valid_count = 0;

        for (const auto& p : trajectory) {
            float dx = p.x - trajectory_center.x;//相对x坐标
            float dy = p.y - trajectory_center.y;//y
            float radius = sqrt(dx*dx + dy*dy);
            float angle = atan2(dy, dx);

            avg_radius += radius;
            avg_angle += angle;
            valid_count++;
        }

        if (valid_count < 2) return predict();//计算平均值

        avg_radius /= valid_count;//半径
        avg_angle /= valid_count;//角度

        // 预测下一帧角度（基于角速度）
        float predicted_angle = avg_angle + angular_velocity * dt * CV_PI / 180.0f;//角度预测
        Point2f predicted_pos = trajectory_center + Point2f(
            cos(predicted_angle) * avg_radius,
            sin(predicted_angle) * avg_radius//当前平均角度
        );

        // 计算预测置信度
        float prediction_error = 0.0f;
        for (const auto& p : trajectory) {
            float dx = p.x - predicted_pos.x;
            float dy = p.y - predicted_pos.y;
            prediction_error += sqrt(dx*dx + dy*dy);
        }
        prediction_error /= trajectory.size();
        prediction_confidence = max(0.3f, min(1.0f, 1.0f - prediction_error / 100.0f));// 置信度计算

        // 结合卡尔曼预测和旋转预测
        Point2f kalman_pred = predict();
        Point2f adaptive_pred = predicted_pos;

        Point2f combined_pred = Point2f(// 加权融合
            kalman_pred.x * (1.0f - adapt_factor) + adaptive_pred.x * adapt_factor,
            kalman_pred.y * (1.0f - adapt_factor) + adaptive_pred.y * adapt_factor
        );

        state.at<float>(0) = combined_pred.x;
        state.at<float>(1) = combined_pred.y;//更新y
        kf.statePost = state;//更新卡尔曼状态

        return combined_pred;
    }

    void reset() {//重置平台
        has_last_pos = false;//位置
        state = Mat::zeros(4, 1, CV_32F);//向量
        setIdentity(kf.errorCovPost, Scalar::all(1));//协方差矩阵
        trajectory.clear();//轨迹清空
        is_spinning = false;//旋转状态
        spin_frame_count = 0;
        angular_velocity = 0.0f;
        use_adaptive_predict = false;//预测
        prediction_confidence = 1.0f;
    }
};//作用 错误恢复 内存管理 模式切换

#endif // RM3_ALGORITHM_H
