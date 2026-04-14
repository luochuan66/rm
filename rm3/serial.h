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
#pragma pack(1)// 数据结构定义强制字节对齐：取消结构体默认的字节对齐 确保数据紧凑：避免填充字节，保证跨平台通信一致性
struct VisionSendData
{
    uint8_t header;//// 帧头：0xA5
    float yaw;//// 偏航角（度）
    float pitch;/// 俯仰角（度）
    uint8_t shoot;// 射击标志：1=射击，0=不射击
    uint8_t checksum;// 校验和
};
#pragma pack()

// ===================== 串口类 =====================
class SerialPort {//串口类成员变量
private:
    int fd;//// 文件描述符
    bool is_opened;//// 串口打开状态标志

public:
    SerialPort() : fd(-1), is_opened(false) {}

    ~SerialPort() {
        close();
    }

    // 打开串口 串口打开配置
    bool open(const string& port = "/dev/ttyUSB0", int baudrate = B115200) {
        fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);//串口参数详解
        if (fd == -1) {
            perror("串口打开失败");
            return false;
        }

        struct termios options;//创建结构体变量
        tcgetattr(fd, &options);//获取当前串口的终端属性

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
        options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);//码禁用了4个终端特性

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
            return -1;//串口未打开，返回错误
        }
        int n = write(fd, data, size);// 系统调用写入数据
        if (n < 0) {
            perror("串口写入失败");// 打印错误信息
        }
        return n;
    }

    // 发送视觉数据
    bool sendVisionData(float yaw, float pitch, bool shoot) {//// 1. 状态检查
        if (!is_opened || fd == -1) {
            return false;
        }

        VisionSendData send_data;
        send_data.header = 0xA5;// 帧头
        send_data.yaw = yaw;// 偏航角
        send_data.pitch = pitch;// 俯仰角
        send_data.shoot = shoot ? 1 : 0; // 射击标志
        send_data.checksum = checkSum((uint8_t*)&send_data, sizeof(send_data)-1);// 3. 计算校验和（排除checksum字段自身）

        int n = send(&send_data, sizeof(send_data));// 4. 发送数据包
        return n == sizeof(send_data); // 5. 验证发送结果
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
