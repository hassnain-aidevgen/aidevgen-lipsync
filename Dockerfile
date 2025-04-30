# Use NVIDIA CUDA base image with cuDNN for PyTorch
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Prevent interactive prompts and optimize pip
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install OS packages and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl wget git ffmpeg libgl1 libsm6 libxext6 unzip \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Install pip and set Python defaults
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy codebase and scripts (excluding models via .dockerignore)
COPY MuseTalk/ MuseTalk/
COPY scripts/ scripts/

# Pre-install requests (needed for model downloader)
RUN pip install requests

# Download models into MuseTalk/models at build time
# RUN python3 scripts/download_all_weights.py

# Install main dependencies
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install -r MuseTalk/requirements.txt && \
    pip install boto3 runpod ffmpeg-python imageio moviepy opencv-python-headless omegaconf tqdm

# Launch RunPod handler
CMD ["python3", "scripts/runpod_handler.py"]
