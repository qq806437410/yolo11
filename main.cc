/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <float.h>
#include <vector>
#include <chrono>
#include "BYTETracker.h"

#include "yolo11.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "opencv2/opencv.hpp"
#include <opencv2/videoio.hpp>

#define model_path "../model/yolo11.rknn"
// 视频路径（本地视频 or RTSP）
#define video_path1 "../model/video7.mp4"
#define video_path2 "../model/video2.mp4"
#define video_path3 "../model/video3.mp4"
// #define video_path1 "rtsp://admin:12345678q@192.168.2.223:554/h264/ch1/main/av_stream"

#define OutScaleX 4
#define OutScaleY 2.25f

// #define OutScaleX 3
// #define OutScaleY 1.6875f

#define NUM_VIDEO_SOURCES 1
// 视频数据队列
std::vector<std::deque<std::tuple<cv::Mat, cv::Mat, int>>> frame_queues(NUM_VIDEO_SOURCES);
std::vector<std::deque<std::tuple<cv::Mat, std::vector<Object>, object_detect_result_list>>> inference_queues(NUM_VIDEO_SOURCES);
// std::vector<std::deque<std::tuple<cv::Mat, std::vector<Object>, object_detect_result_list>>> postprocess_queues(NUM_VIDEO_SOURCES);
std::vector<std::deque<cv::Mat>> postprocess_queues(NUM_VIDEO_SOURCES);
// 互斥锁（每个视频源一个锁，简化管理）
std::mutex queue_mutex[NUM_VIDEO_SOURCES];

// 条件变量
std::condition_variable queue_cv_readerAndInfer[NUM_VIDEO_SOURCES];
std::condition_variable queue_cv_InferAndPost[NUM_VIDEO_SOURCES];
std::condition_variable queue_cv_PostAndWrite[NUM_VIDEO_SOURCES];

bool stop_processing = false;

// 在 mat_to_image_buffer 函数中，增加对 malloc 的返回值检查
void mat_to_image_buffer(const cv::Mat &frame, image_buffer_t *img)
{
    if (!img)
        return;

    // 设置宽度和高度
    img->width = frame.cols;
    img->height = frame.rows;

    // 计算 stride
    img->width_stride = frame.cols * 3; // 3 通道（RGB888）
    img->height_stride = frame.rows;

    img->format = IMAGE_FORMAT_RGB888;
    img->size = img->width_stride * img->height_stride; // 总大小

    // 为图像数据分配内存
    img->virt_addr = (unsigned char *)malloc(img->size);

    if (!img->virt_addr)
    {
        std::cerr << "Failed to allocate memory for image buffer" << std::endl;
        return;
    }

    // OpenCV 默认是 BGR 格式，转换为 RGB
    cv::Mat rgb_img;
    cv::cvtColor(frame, rgb_img, cv::COLOR_BGR2RGB);

    // 拷贝数据到 `virt_addr`
    memcpy(img->virt_addr, rgb_img.data, img->size);
}

void video_reader(int video_id)
{
    cv::Mat frame;
    cv::VideoCapture cap_video;
    cv::Mat resized_frame;
    cv::Mat final_frame;

    int original_height = 1440;
    int original_width = 2560;
    // 目标大小为640x640
    int target_width = 640;
    int target_height = 640;
    int new_width;
    int new_height;

    float scale_x;
    float scale_y;

    int frame_id = 0;

    // 计算宽度和高度的缩放因子
    scale_x = (float)target_width / original_width;
    scale_y = (float)target_height / original_height;

    // 计算新的缩放后的图像尺寸
    new_width = (int)(original_width * scale_x);
    new_height = (int)(original_height * scale_y);

    // 计算裁剪区域的起始位置，以确保中心对齐
    int start_x = (target_width - new_width) / 2;
    int start_y = (target_height - new_height) / 2;

    // 根据 video_id 选择视频源
    if (video_id == 0)
        cap_video.open(video_path1);
    else if (video_id == 1)
        cap_video.open(video_path2);
    else if (video_id == 2)
        cap_video.open(video_path3);
    else
    {
        std::cerr << "Error: Invalid video_id: " << video_id << std::endl;
        return;
    }

    // 检查视频流是否成功打开
    if (!cap_video.isOpened())
    {
        std::cerr << "Error: Failed to open video stream for video_id: " << video_id << std::endl;
        return;
    }
    // 获取视频的帧率
    double fps = cap_video.get(cv::CAP_PROP_FPS);
    std::cout << "视频帧率: " << fps << " FPS" << std::endl;
    while (!stop_processing)
    {
        auto start_time = std::chrono::high_resolution_clock::now(); // 记录循环开始时间
        cap_video >> frame;

        if (frame.empty())
            break;

        // cv::Mat frame_copy = cv::Mat::zeros(frame.size(), frame.type());
        // frame.copyTo(frame_copy); // 复制frame的内容到frame_copy

        cv::resize(frame, resized_frame, cv::Size(new_width, new_height));
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(); // 计算耗时（微秒）
        // std::cout << "ReadVideo loop time for video_id " << video_id << ": " << duration << " microseconds." << std::endl;
        // // 创建一个黑色背景
        // final_frame = cv::Mat::zeros(target_height, target_width, frame.type());

        // // 将缩放后的图像放到背景上，居中显示
        // resized_frame.copyTo(final_frame(cv::Rect(start_x, start_y, new_width, new_height)));
        frame_id++;
        {
            std::lock_guard<std::mutex> lock(queue_mutex[video_id]);
            if (frame_queues[video_id].size() >= 1000)
            {
                frame_queues[video_id].pop_back(); // 删除最旧的帧
            }
            frame_queues[video_id].push_front({resized_frame, frame, frame_id});
        }
        queue_cv_readerAndInfer[video_id].notify_all();
    }

    stop_processing = true;
}

void inference(int video_id, rknn_app_context_t &rknn_app_ctx)
{
    int ret = init_yolo11_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        std::cerr << "YOLO 初始化失败！" << std::endl;
        return;
    }
    int frame_id;
    std::vector<Object> objects;
    cv::Mat frame, resized_frame;
    object_detect_result_list od_results;
    image_buffer_t src_image;
    rknn_perf_run perf_info;

    while (!stop_processing)
    {
        auto start_time = std::chrono::high_resolution_clock::now(); // 记录循环开始时间
        // 读取推理队列
        {
            std::unique_lock<std::mutex> lock(queue_mutex[video_id]);
            if (!queue_cv_readerAndInfer[video_id].wait_for(lock, std::chrono::milliseconds(0),
                                                            [&]()
                                                            { return !frame_queues[video_id].empty(); }))
            {
                // std::cerr << "推理队列超时!" << std::endl;
                continue;
            }
            auto &data = frame_queues[video_id].front();

            lock.unlock(); // 提前释放锁，提高并发性

            // data.first.copyTo(resized_frame);
            // data.second.copyTo(frame);
            resized_frame = std::get<0>(data);
            frame = std::get<1>(data);
            frame_id = std::get<2>(data);
            frame_queues[video_id].pop_back(); // 先取出数据再弹出
        }

        // 推理处理
        mat_to_image_buffer(resized_frame, &src_image);
        ret = inference_yolo11_model(&rknn_app_ctx, &src_image, &od_results);
        free(src_image.virt_addr); // 释放分配的内存

        if (ret != 0)
        {
            std::cerr << "YOLO 推理失败！" << std::endl;
            continue;
        }
        // std::cout << "线程 " << std::this_thread::get_id() << " 处理帧: " << frame_id << std::endl;

        // ret = rknn_query(rknn_app_ctx.rknn_ctx, RKNN_QUERY_PERF_RUN, &perf_info, sizeof(perf_info));
        // if (ret == RKNN_SUCC)
        // {
        //     // std::cout << "Processing frame in thread: " << std::this_thread::get_id() << std::endl;
        //     // std::cout << "thread :" << video_id << "Inference time (us): " << perf_info.run_duration << std::endl;
        // }
        // else
        // {
        //     std::cerr << "Failed to get performance info!" << std::endl;
        // }

        // 解析推理结果
        for (int i = 0; i < od_results.count; i++)
        {
            object_detect_result det = od_results.results[i];
            Object obj;
            obj.label = det.cls_id;
            obj.prob = det.prop;
            obj.rect = cv::Rect_<float>(
                det.box.left * OutScaleX,
                det.box.top * OutScaleY,
                (det.box.right - det.box.left) * OutScaleX,
                (det.box.bottom - det.box.top) * OutScaleY);
            objects.push_back(obj);
        }

        // 存入后处理队列
        {
            std::lock_guard<std::mutex> lock(queue_mutex[video_id]);

            if (inference_queues[video_id].size() >= 1000)
            {
                inference_queues[video_id].pop_back(); // 删除最旧的帧
            }
            inference_queues[video_id].push_front({frame, objects, od_results});
            // std::cout << "Pushing data to postprocess_queues" << std::endl;
        }

        queue_cv_InferAndPost[video_id].notify_all();
        objects.clear();

        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count(); // 计算耗时（微秒）

        // // 输出每轮循环的耗时
        // std::cout << "Inference loop time for video_id " << video_id << ": " << duration << " microseconds." << std::endl;
    }

    release_yolo11_model(&rknn_app_ctx);
}

void postprocess(int video_id, BYTETracker &tracker)
{
    cv::Mat frame;
    static int i = 0;
    // 在循环外部定义变量
    int frame_count = 0;
    double time_prev = (double)cv::getTickCount();
    std::string filename = "/home/aukun/Project-AI/AI-ENV/rknn_model_zoo-main/examples/yolo11/model/output_video_0.avi";
    // cv::VideoWriter out(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(2560, 1440));

    std::vector<STrack> output_stracks;
    std::vector<Object> objects;
    object_detect_result_list od_results;

    while (!stop_processing)
    {
        std::tuple<cv::Mat, std::vector<Object>, object_detect_result_list> data;
        {
            std::unique_lock<std::mutex> lock(queue_mutex[video_id]);
            queue_cv_InferAndPost[video_id].wait(lock, [video_id]
                                                 { return !inference_queues[video_id].empty() || stop_processing; });

            if (stop_processing && inference_queues[video_id].empty())
                break;

            data = inference_queues[video_id].front();
            inference_queues[video_id].pop_front();
        }
        std::get<0>(data).copyTo(frame);
        // frame = std::get<0>(data);
        objects = std::get<1>(data);
        od_results = std::get<2>(data);

        // 后处理代码...
        if (frame.empty())
        {
            std::cerr << "postprocess Error: Empty frame detected!" << std::endl;
            continue;
        }

        // 目标跟踪
        output_stracks = tracker.update(objects);

        //  objects 是目标检测后的对象集合
        for (size_t i = 0; i < od_results.count; ++i)
        {
            int r = rand() % 256;
            int g = rand() % 256;
            int b = rand() % 256;
            Scalar randomColor(r, g, b);
            // 使用跟踪的 ID
            object_detect_result *det_result = &(od_results.results[i]);
            int x1 = (int)(det_result->box.left * OutScaleX);
            int y1 = (int)(det_result->box.top * OutScaleY);
            int x2 = (int)(det_result->box.right * OutScaleX);
            int y2 = (int)(det_result->box.bottom * OutScaleY);

            // 绘制矩形框
            cv::rectangle(frame, objects[i].rect, randomColor, 2);

            // 显示对应的 ID
            char text[256];
            sprintf(text, " %.1f%%", od_results.results->prop * 100);
            cv::putText(frame, text, cv::Point(x1, y1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, randomColor, 2);
        }

        // 绘制跟踪框
        for (int i = 0; i < output_stracks.size(); i++)
        {
            vector<float> tlwh = output_stracks[i].tlwh;
            tlwh[0] = tlwh[0];
            tlwh[1] = tlwh[1];
            tlwh[2] = tlwh[2];
            tlwh[3] = tlwh[3];

            bool vertical = tlwh[2] / tlwh[3] > 1.6;
            if (tlwh[2] * tlwh[3] > 20 && !vertical)
            {
                // 绘制跟踪框，并标注Tracker ID
                putText(frame, format("Tracker ID: %d", output_stracks[i].track_id), Point(tlwh[0] + 10, tlwh[1] - 5),
                        0, 0.6, Scalar(16, 119, 0), 2, LINE_AA);
            }
        }

        // cout << "size" << frame.cols << "x" << frame.rows << endl;
        // 显示结果
        // cv::namedWindow("RTSP Stream", cv::WINDOW_NORMAL);
        // cv::resizeWindow("RTSP Stream", 1920, 1080);

        if (cv::waitKey(30) == 'q')
        {
            stop_processing = true;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex[video_id]);

            if (postprocess_queues[video_id].size() >= 1000)
            {
                postprocess_queues[video_id].pop_back(); // 删除最旧的帧
            }
            postprocess_queues[video_id].push_front(frame);
        }
        queue_cv_PostAndWrite[video_id].notify_all();

        output_stracks.clear();
        objects.clear();
    }
}
void video_write_worker()
{
    int video_id = 0;
    std::string filename = "../model/output_video_0.avi";
    cv::VideoWriter out(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 50, cv::Size(2560, 1440));
    // cv::VideoWriter out(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(1920, 1080));

    cv::Mat frame;
    while (!stop_processing)
    {

        {
            // std::cout << "Queue size: " << postprocess_queues[video_id].size() << std::endl;
            std::unique_lock<std::mutex> lock(queue_mutex[video_id]);
            if (!queue_cv_PostAndWrite[video_id].wait_for(lock, std::chrono::milliseconds(50),
                                                          [&]()
                                                          { return !postprocess_queues[video_id].empty(); }))
            {
                // std::cerr << "写队列超时!" << std::endl;
                continue;
            }

            frame = postprocess_queues[video_id].front();
            postprocess_queues[video_id].pop_back();
        }
        cv::moveWindow("RTSP Stream", 0, 0);
        cv::imshow("RTSP Stream", frame);
        waitKey(1);
        out.write(frame); // 写入视频帧
    }
}

int main()
{
    // XInitThreads(); // 初始化 X11 的多线程支持
    // 启动每个视频源的线程
    rknn_app_context_t rknn_app_ctx_0, rknn_app_ctx_1, rknn_app_ctx_2;
    BYTETracker tracker_0(20, 30), tracker_1(20, 30), tracker_2(20, 30);

    std::thread thread_0_video_reader(video_reader, 0);

    std::thread thread_0_inference(inference, 0, std::ref(rknn_app_ctx_0));
    // std::thread thread_0_inference_1(inference, 0, std::ref(rknn_app_ctx_1));
    // std::thread thread_0_inference_2(inference, 0, std::ref(rknn_app_ctx_2));

    std::thread thread_0_postprocess(postprocess, 0, std::ref(tracker_0));
    std::thread thread_0_postprocess_writer(video_write_worker);

    // 等待所有线程完成
    thread_0_video_reader.join();

    thread_0_inference.join();
    // thread_0_inference_1.join();
    // thread_0_inference_2.join();

    thread_0_postprocess.join();

    thread_0_postprocess_writer.join();

    return 0;
}
