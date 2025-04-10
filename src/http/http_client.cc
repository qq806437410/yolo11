#include "http/http_client.h"

// 构造函数，初始化服务器 URL
HttpClient::HttpClient(const std::string &server_url) : server_url_(server_url)
{
    this->devID = "6000001";
    this->devType = "null";
    this->objID = "null";
    this->channelID = "null";
    this->realTime = "null";
    this->counterTime = "null";
    this->counterType = "null";
    this->counterID = "null";
    this->trackTime = "null";
    this->receptionTime = "null";
    this->inNum = "null";
    this->outNum = "null";
    this->passNum = "null";
    this->bodyPictureCrc32 = "null";
    this->age = "null";
    this->gender = "null";
}

// 读取图像并转换为 Base64 编码
std::string HttpClient::imageToBase64(const std::string &image_path)
{
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file)
    {
        std::cerr << "无法打开图片文件: " << image_path << std::endl;
        return "";
    }

    std::ostringstream oss;
    oss << image_file.rdbuf();
    std::string image_data = oss.str();

    // 使用 httplib 的 base64 编码
    return httplib::detail::base64_encode(image_data); // 将 image_data 直接传递给 base64_encode
}

std::string HttpClient::matToBase64(const cv::Mat &image, const std::string &format)
{
    std::vector<uchar> buf;
    std::vector<int> param = {cv::IMWRITE_JPEG_QUALITY, 90}; // 可调节质量

    if (!cv::imencode(format, image, buf, param))
    {
        std::cerr << "Error: Image encoding failed." << std::endl;
        return "";
    }

    // 直接使用 httplib 提供的 Base64 编码
    return httplib::detail::base64_encode(std::string(buf.begin(), buf.end()));
}
// 发送 HTTP 请求
void HttpClient::sendData(const std::string &image_base64, const std::vector<double> &bodyFeatures)
{
    // 构造 JSON 数据
    json payload;
    payload["devID"] = this->devID;
    payload["objID"] = this->objID;
    payload["channelID"] = this->channelID;
    payload["realTime"] = this->realTime;
    payload["counterTime"] = this->counterTime;
    payload["counterType"] = this->counterType;
    payload["counterID"] = this->counterID;
    payload["trackTime"] = this->trackTime;
    payload["receptionTime"] = this->receptionTime;
    payload["inNum"] = this->inNum;
    payload["outNum"] = this->outNum;
    payload["passNum"] = passNum;
    payload["bodyPicture"] = image_base64;
    payload["bodyPictureCrc32"] = this->bodyPictureCrc32;
    payload["features"] = bodyFeatures;
    payload["gender"] = this->gender;
    payload["age"] = this->age;

    // 打印 JSON 数据，确保它的格式正确
    // std::cout << "发送的 JSON 数据: " << payload.dump(4) << std::endl;

    // 创建客户端对象
    httplib::Client cli(server_url_);

    // 设置头部信息，指定内容类型为 application/json
    httplib::Headers headers = {
        {"Content-Type", "application/json"}};

    // 发送 POST 请求，附带 JSON 数据
    auto res = cli.Post("/upload", headers, payload.dump(), "application/json");

    // 检查是否成功发起请求
    if (!res)
    {
        std::cerr << "请求失败，错误代码: " << res.error() << std::endl;
        return;
    }

    // 打印响应状态和响应体
    std::cout << "响应状态: " << res->status << std::endl;
    std::cout << "响应内容: " << res->body << std::endl;
}
