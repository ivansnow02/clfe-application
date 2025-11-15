FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 替换为阿里云镜像源
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 基本工具和系统依赖(使用系统自带的 Python 3.8)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    ffmpeg \
    libopenblas-dev \
    libboost-all-dev \
    python3 python3-venv python3-dev python3-pip \
    git curl wget \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 OpenCV 4.1.0 (OpenFace 需要的版本)
RUN apt-get update && apt-get install -y \
    cmake build-essential unzip \
    && cd /tmp \
    && wget -q https://github.com/opencv/opencv/archive/4.1.0.zip -O opencv.zip \
    && unzip opencv.zip \
    && cd opencv-4.1.0 \
    && mkdir build && cd build \
    && cmake -D CMAKE_BUILD_TYPE=Release \
             -D CMAKE_INSTALL_PREFIX=/usr/local \
             -D BUILD_TESTS=OFF \
             -D BUILD_PERF_TESTS=OFF \
             -D BUILD_EXAMPLES=OFF \
             -D WITH_TBB=ON \
             -D WITH_V4L=ON \
             .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd / && rm -rf /tmp/opencv* \
    && apt-get remove -y cmake build-essential wget unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 20.04 自带 Python 3.8，足够运行 MMSA-FET
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# 拷贝当前项目代码进容器
COPY . /workspace

# Python 依赖（CPU 版），使用清华镜像源
RUN python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    # pip install "numpy<2" scikit-learn einops transformers tensorboardX pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install MMSA-FET -i https://pypi.tuna.tsinghua.edu.cn/simple

# 预下载 MMSA-FET 的模型和资源（失败也不中断）
RUN python -m MSA_FET install || true
