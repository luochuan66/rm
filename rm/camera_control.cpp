#include "camera_control.h"
#include <iostream>
#include <cstring>
#include <iomanip>

using namespace std;

HikCamera::HikCamera() : handle(NULL), is_initialized(false) {}

HikCamera::~HikCamera() {
    release();
}

bool HikCamera::init() {
    unsigned int nTLayerType = MV_GIGE_DEVICE | MV_USB_DEVICE;
    MV_CC_DEVICE_INFO_LIST stDeviceList;
    memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    // 枚举设备
    int nRet = MV_CC_EnumDevices(nTLayerType, &stDeviceList);
    if (nRet != MV_OK || stDeviceList.nDeviceNum == 0) {
        cout << "未找到海康相机！" << endl;
        return false;
    }

    cout << "找到 " << stDeviceList.nDeviceNum << " 个相机设备" << endl;

    // 选择第一个设备
    int nIndex = 0;
    MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[nIndex];

    // 创建句柄
    nRet = MV_CC_CreateHandle(&handle, pDeviceInfo);
    if (nRet != MV_OK) {
        cout << "创建相机句柄失败！" << endl;
        return false;
    }

    // 打开设备
    nRet = MV_CC_OpenDevice(handle);
    if (nRet != MV_OK) {
        cout << "打开相机失败！" << endl;
        MV_CC_DestroyHandle(handle);
        handle = NULL;
        return false;
    }

    cout << "相机打开成功！" << endl;

    // 设置触发模式为关闭
    MV_CC_SetEnumValue(handle, "TriggerMode", MV_TRIGGER_MODE_OFF);

    // 设置像素格式为 RGB8（彩色）
    MV_CC_SetEnumValue(handle, "PixelFormat", PixelType_Gvsp_RGB8_Packed);

    // 设置曝光时间
    MV_CC_SetFloatValue(handle, "ExposureTime", 3000.0);

    // 设置增益
    MV_CC_SetFloatValue(handle, "Gain", 0.0);

    // 读取并设置为相机最大帧率
    MVCC_FLOATVALUE stFloatValue;
    float max_frame_rate = 30.0f;  // 默认值

    // 读取帧率范围
    nRet = MV_CC_GetFloatValue(handle, "AcquisitionFrameRate", &stFloatValue);
    if (nRet == MV_OK) {
        max_frame_rate = stFloatValue.fMax;  // 获取最大帧率
        cout << "相机最大帧率: " << fixed << setprecision(1) << max_frame_rate << " fps" << endl;

        // 设置为最大帧率
        nRet = MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", max_frame_rate);
        if (nRet == MV_OK) {
            cout << "已设置为最大帧率: " << fixed << setprecision(1) << max_frame_rate << " fps" << endl;
        } else {
            cout << "设置帧率失败！错误码: 0x" << hex << nRet << endl;
        }
    } else {
        cout << "无法读取帧率范围，使用默认 30fps" << endl;
        MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", 30.0);
    }

    is_initialized = true;
    return true;
}

void HikCamera::release() {
    if (handle != NULL) {
        stopGrabbing();
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
        handle = NULL;
        is_initialized = false;
    }
}

bool HikCamera::startGrabbing() {
    if (!is_initialized || handle == NULL) {
        return false;
    }
    int nRet = MV_CC_StartGrabbing(handle);
    if (nRet != MV_OK) {
        cout << "开始取流失败！" << endl;
        return false;
    }
    cout << "相机取流已启动！" << endl;
    return true;
}

void HikCamera::stopGrabbing() {
    if (handle != NULL) {
        MV_CC_StopGrabbing(handle);
    }
}

int HikCamera::getOneFrame(unsigned char* pData, unsigned int nDataSize, MV_FRAME_OUT_INFO_EX* pFrameInfo, int nMsec) {
    if (!is_initialized || handle == NULL) {
        return -1;
    }
    return MV_CC_GetOneFrameTimeout(handle, pData, nDataSize, pFrameInfo, nMsec);
}

int HikCamera::setExposureTime(float exposureTime) {
    if (!is_initialized || handle == NULL) {
        return -1;
    }
    return MV_CC_SetFloatValue(handle, "ExposureTime", exposureTime);
}

int HikCamera::setGain(float gain) {
    if (!is_initialized || handle == NULL) {
        return -1;
    }
    return MV_CC_SetFloatValue(handle, "Gain", gain);
}

int HikCamera::setPixelFormat(int pixelFormat) {
    if (!is_initialized || handle == NULL) {
        return -1;
    }
    return MV_CC_SetEnumValue(handle, "PixelFormat", pixelFormat);
}
