#include "serial_port.h"
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

using namespace std;

SerialPort::SerialPort() : fd(-1), is_opened(false) {}

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const string& port, int baudrate) {
    fd = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        perror("串口打开失败");
        return false;
    }

    struct termios options;
    tcgetattr(fd, &options);

    // 设置波特率
    speed_t speed;
    switch (baudrate) {
        case 9600:   speed = B9600; break;
        case 19200:  speed = B19200; break;
        case 38400:  speed = B38400; break;
        case 57600:  speed = B57600; break;
        case 115200: speed = B115200; break;
        default:     speed = B9600; break;
    }
    cfsetispeed(&options, speed);
    cfsetospeed(&options, speed);

    // 配置串口参数
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CRTSCTS;
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    tcsetattr(fd, TCSANOW, &options);
    is_opened = true;
    return true;
}

void SerialPort::close() {
    if (fd != -1) {
        ::close(fd);
        fd = -1;
        is_opened = false;
    }
}

bool SerialPort::isOpen() const {
    return is_opened;
}

int SerialPort::writeData(const void* data, size_t len) {
    if (!is_opened || fd == -1) {
        return -1;
    }
    return ::write(fd, data, len);
}

uint8_t SerialPort::checkSum(uint8_t* data, int len) {
    uint8_t sum = 0;
    for (int i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum;
}
