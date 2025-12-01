# CLFE - 跨语言特征提取与情感分析服务

基于 ALMT 模型的多模态情感分析服务，支持视频输入的音频、视觉、文本特征提取与情感预测。

## 架构概述

项目包含两个 Docker 服务：

| 服务           | 端口 | 描述                                     |
| -------------- | ---- | ---------------------------------------- |
| `clfe-fet-api` | 8000 | 特征提取服务（音视频处理、ASR 语音识别） |
| `clfe-service` | 8001 | 情感预测服务（ALMT 模型推理）            |


## 前置要求

- Docker >= 20.10
- Docker Compose >= 2.0
- NVIDIA GPU + 驱动
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 快速开始

### 1. 创建必要的目录

```bash
# 创建模型缓存目录
mkdir -p .cache/huggingface/transformers

# 创建检查点目录（如果不存在）
mkdir -p ckpt
```

### 2. 检查 CUDA 版本

```bash
nvidia-smi
```

查看输出中的 `CUDA Version`，选择对应的 PyTorch CUDA 版本。


### 3. 设置 CUDA 版本并启动

| CUDA 版本 | 环境变量值      |
| --------- | --------------- |
| CUDA 11.8 | `cu118`         |
| CUDA 12.1 | `cu121`         |
| CUDA 12.4 | `cu124`         |
| CUDA 12.8 | `cu128`（默认） |

**方式一：使用默认 CUDA 12.8**

```bash
docker compose up -d --build
```

**方式二：通过命令行指定 CUDA 版本**

```bash
# 例如使用 CUDA 12.1
CUDA_VERSION=cu121 docker compose up -d --build
```



## 环境变量配置

### docker-compose.yml 中的服务配置

| 变量                   | 默认值                     | 描述                           |
| ---------------------- | -------------------------- | ------------------------------ |
| `CUDA_VERSION`         | `cu128`                    | PyTorch CUDA 版本              |
| `TOOLS_SERVER`         | `http://clfe-fet-api:8000` | 特征提取服务地址               |
| `MAX_CONCURRENT_TASKS` | `2`                        | 最大并发任务数（根据显存调整） |

### 调整并发数

根据 GPU 显存大小调整 `MAX_CONCURRENT_TASKS`：

```yaml
environment:
  - MAX_CONCURRENT_TASKS=1  # 根据显存调整
```

## 数据卷说明

| 挂载路径                            | 容器路径    | 描述                 |
| ----------------------------------- | ----------- | -------------------- |
| `./.cache/huggingface/transformers` | `/models`   | HuggingFace 模型缓存 |
| `./ckpt`                            | `/app/ckpt` | 模型检查点目录       |
