FROM python:3.11-bullseye
ARG CUDA_VERSION=cu128

# 换源（中科大）
RUN sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org|mirrors.ustc.edu.cn/debian-security|g' /etc/apt/sources.list

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optionally add ffmpeg if needed for video preview)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models
# Copy project files
COPY pyproject.toml .
# 升级 pip 并配置源（中科大）
RUN pip install --upgrade pip && \
    pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
# Use pip for now (no build backend specified)

RUN pip install --no-cache-dir torch==2.9.1 torchvision torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/${CUDA_VERSION}

RUN pip install --upgrade pip && \
    pip install .

COPY src ./src
COPY configs ./configs

# Default env pointing to SIMS config and a checkpoint; override as needed
ENV CLFE_CONFIG=configs/sims.yaml \
    CLFE_CKPT=ckpt/best_binary_epoch_5.pth \
    API_PORT=8001 \
    HF_ENDPOINT=https://hf-mirror.com \
    TRANSFORMERS_CACHE=/models 

EXPOSE 8001

CMD ["python", "-m", "src.clfe.api"]
