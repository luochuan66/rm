#ifndef CAMERA_CONTROL_H
#define CAMERA_CONTROL_H

// 必须先包含海康SDK，避免与标准库冲突
#include "MvCameraControl.h"

#include <string>

class HikCamera {
private:
    void* handle;
    bool is_initialized;

public:
    HikCamera();
    ~HikCamera();
    
    bool init();
    void release();
    bool startGrabbing();
    void stopGrabbing();
    int getOneFrame(unsigned char* pData, unsigned int nDataSize, MV_FRAME_OUT_INFO_EX* pFrameInfo, int nMsec);
    int setExposureTime(float exposureTime);
    int setGain(float gain);
    int setPixelFormat(int pixelFormat);
};

#endif // CAMERA_CONTROL_H
