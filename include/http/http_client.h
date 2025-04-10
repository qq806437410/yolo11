#ifndef HTTP_CLIENT_H
#define HTTP_CLIENT_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "httplib.h"
#include <json.hpp>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

class HttpClient
{
public:
    // 构造函数，传入服务器 URL
    HttpClient(const std::string &server_url);

    // 读取图像并转换为 Base64 编码
    std::string imageToBase64(const std::string &image_path);
    std::string matToBase64(const cv::Mat &image, const std::string &format = ".jpg");

    // 发送 HTTP 请求
    void sendData(const std::string &image_base64, const std::vector<double> &features);

private:
    std::string server_url_; // 存储服务器 URL

    std::string devID;
    std::string devType;
    std::string objID;
    std::string channelID;
    std::string realTime;
    std::string counterTime;
    std::string counterType;
    std::string counterID;
    std::string trackTime;
    std::string receptionTime;
    std::string inNum;
    std::string outNum;
    std::string passNum;
    std::string bodyPictureCrc32;
    std::string gender;
    std::string age;
};

#endif // HTTP_CLIENT_H
