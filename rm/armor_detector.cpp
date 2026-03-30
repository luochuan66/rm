#include "armor_detector.h"
#include <opencv2/calib3d.hpp>
#include <cmath>

using namespace cv;
using namespace std;

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
