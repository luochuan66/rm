#ifndef RM3_SERIAL_H
#define RM3_SERIAL_H

#include <iostream>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <cstring>

using namespace std;

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

// ===================== 串口类 =====================
class SerialPort {
private:
    int fd;
    bool is_opened;

public:
    SerialPort() : fd(-1), is_opened(false) {}

    ~SerialPort() {
        close();
    }

    // 打开串口
    bool open(const string& port = "/dev/ttyUSB0", int baudrate = B115200) {
        fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd == -1) {
            perror("串口打开失败");
            return false;
        }

        struct termios options;
        tcgetattr(fd, &options);

        // 设置波特率
        cfsetispeed(&options, baudrate);
        cfsetospeed(&options, baudrate);

        // 设置数据位、停止位、校验位
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~CSIZE;
        options.c_cflag |= CS8;              // 8数据位
        options.c_cflag &= ~PARENB;          // 无校验位
        options.c_cflag &= ~CSTOPB;          // 1停止位
        options.c_cflag &= ~CRTSCTS;         // 无硬件流控

        // 设置输入模式
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

        // 设置输出模式
        options.c_oflag &= ~OPOST;

        // 应用设置
        tcsetattr(fd, TCSANOW, &options);

        is_opened = true;
        cout << "串口 " << port << " 打开成功" << endl;
        return true;
    }

    // 关闭串口
    void close() {
        if (is_opened && fd != -1) {
            ::close(fd);
            fd = -1;
            is_opened = false;
            cout << "串口已关闭" << endl;
        }
    }

    // 发送数据
    int send(const void* data, size_t size) {
        if (!is_opened || fd == -1) {
            return -1;
        }
        int n = write(fd, data, size);
        if (n < 0) {
            perror("串口写入失败");
        }
        return n;
    }

    // 发送视觉数据
    bool sendVisionData(float yaw, float pitch, float distance, bool shoot) {
        if (!is_opened || fd == -1) {
            return false;
        }

        VisionSendData send_data;
        send_data.header = 0xA5;
        send_data.yaw = yaw;
        send_data.pitch = pitch;
        send_data.distance = distance;
        send_data.shoot = shoot ? 1 : 0;
        send_data.checksum = checkSum((uint8_t*)&send_data, sizeof(send_data)-1);

        int n = send(&send_data, sizeof(send_data));
        return n == sizeof(send_data);
    }

    // 计算校验和
    static uint8_t checkSum(uint8_t* data, int len) {
        uint8_t sum = 0;
        for (int i = 0; i < len; i++) {
            sum += data[i];
        }
        return sum;
    }

    // 检查是否已打开
    bool isOpen() const {
        return is_opened;
    }
};

#endif // RM3_SERIAL_H
