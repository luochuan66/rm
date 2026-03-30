#ifndef SERIAL_PORT_H
#define SERIAL_PORT_H

#include <string>
#include <cstdint>

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

class SerialPort {
private:
    int fd;
    bool is_opened;

public:
    SerialPort();
    ~SerialPort();
    
    bool open(const std::string& port, int baudrate = 115200);
    void close();
    bool isOpen() const;
    int writeData(const void* data, size_t len);
    static uint8_t checkSum(uint8_t* data, int len);
};

#endif // SERIAL_PORT_H
