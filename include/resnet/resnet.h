#ifndef _RKNN_DEMO_RESNET_H_
#define _RKNN_DEMO_RESNET_H_

#include "rknn_api.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm> // 提供 std::max_element
#include <vector>    // Include this header for std::vector
#include <string>    // Include this header for std::string
#include "yolo11.h"
// typedef struct
// {
//     rknn_context rknn_ctx;
//     rknn_input_output_num io_num;
//     rknn_tensor_attr *input_attrs;
//     rknn_tensor_attr *output_attrs;
//     int model_channel;
//     int model_width;
//     int model_height;
// } rknn_app_context_t;

typedef struct
{
    int cls;
    float score;
} resnet_result;

typedef struct
{
    std::string gender_cls;
    std::string age_cls;
    float gender_score;
    float age_score;
} resnet_parse_result;

int init_resnet_model(const char *model_path, rknn_app_context_t *app_ctx);

int release_resnet_model(rknn_app_context_t *app_ctx);

int inference_resnet_model(rknn_app_context_t *app_ctx, image_buffer_t *img, resnet_result *out_result, int topK);

// 或者更好的方式是直接使用 std:: 前缀（推荐）

// Function declarations using vector
std::vector<std::string> load_labels(const std::string &file_path);
std::vector<float> softmax(const std::vector<float> &logits);

int process_output(resnet_result *out_result, resnet_parse_result *parse_result);

#endif //_RKNN_DEMO_RESNET_H_