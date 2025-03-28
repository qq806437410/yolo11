cmake_minimum_required(VERSION 3.10)
project(yolo11_project)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)  # 或 C++14 / C++17

# 启用AddressSanitizer（ASAN）调试模式
message(STATUS "BUILD WITH ADDRESS SANITIZER")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

# 设置头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/BYTETracker
    ${PROJECT_SOURCE_DIR}/include/eigen3
    ${PROJECT_SOURCE_DIR}/include/opencv4
    ${PROJECT_SOURCE_DIR}/include/rga
    ${PROJECT_SOURCE_DIR}/include/rknpu
    ${PROJECT_SOURCE_DIR}/include/timer
    ${PROJECT_SOURCE_DIR}/include/yolo11
    ${PROJECT_SOURCE_DIR}/utils
)

# 设置库文件路径
set(RKNN_RT_LIB ${PROJECT_SOURCE_DIR}/lib/rknpu/librknnrt.so)
set(RGA_LIB ${PROJECT_SOURCE_DIR}/lib/rknpu/librga.so)
set(RKNN_API_LIB ${PROJECT_SOURCE_DIR}/lib/rknpu/librknn_api.so)
# 自动获取 lib 目录下的所有 .so 文件
file(GLOB_RECURSE OpenCV_LIBS ${PROJECT_SOURCE_DIR}/lib/opencv/*.so)
set(LIBRARY_PATH ${PROJECT_SOURCE_DIR}/lib/utils)


# 添加utils子项目
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)

# 添加源文件
file(GLOB YOLO11_SRC 
    src/yolo11/*.cc
    src/BYTETracker/*.cc
)

# 编译可执行文件
add_executable(yolo11 main.cc ${YOLO11_SRC})

# 链接库
target_link_libraries(yolo11
    imageutils
    fileutils
    imagedrawing 
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
    ${RGA_LIB}
    ${RKNN_API_LIB}

    -lpthread   # 线程支持
    ${LIBRARY_PATH}/libaudioutils.a
    ${LIBRARY_PATH}/libfileutils.a
    ${LIBRARY_PATH}/libimagedrawing.a
    ${LIBRARY_PATH}/libimageutils.a
)

# 设置链接目录
link_directories(${LIBRARY_PATH})
link_directories(${PROJECT_SOURCE_DIR}/lib/opencv)