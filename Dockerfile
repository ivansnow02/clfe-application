# syntax=docker/dockerfile:1
FROM python:3.10-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optionally add ffmpeg if needed for video preview)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
# Use pip for now (no build backend specified)
RUN pip install --upgrade pip && \
    pip install .

COPY src ./src
COPY notebook_configs ./notebook_configs
COPY ckpt ./ckpt

# Default env pointing to SIMS config and a checkpoint; override as needed
ENV CLFE_CONFIG=notebook_configs/sims.yaml \
    CLFE_CKPT=ckpt/ALMT_Demo_SIMS/best_binary_epoch_5.pth \
    TOOLS_SERVER=http://tools:8000 \
    GRADIO_PORT=7860

EXPOSE 7860

CMD ["python", "-m", "src.clfe.gradio_app"]
