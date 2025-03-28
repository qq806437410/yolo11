#!/bin/bash

# 设置 CMake 和构建目录
BUILD_DIR=build
SOURCE_DIR=$(pwd)

# 设置构建目录

if [ ! -d "$BUILD_DIR" ]; then
    echo "cmake has been generated..."
    mkdir -p "$BUILD_DIR"
fi

# 进入构建目录
cd "$BUILD_DIR"

# 如果 CMakeCache.txt 不存在，则运行 CMake 配置
if [ ! -f "CMakeCache.txt" ]; then
    echo "Configuring the project with CMake..."
    cmake "$SOURCE_DIR" -DCMAKE_BUILD_TYPE=Release
else
    echo "CMake configuration already exists, skipping..."
fi

# 编译项目
echo "Building the project..."
make -j4

# 运行生成的可执行文件
echo "Running the executable..."
cp yolo11 $SOURCE_DIR

# 返回到源目录
cd $SOURCE_DIR
