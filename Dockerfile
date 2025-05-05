# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Disable prompts and cache
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install OS packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl wget git unzip ffmpeg \
        libgl1 libsm6 libxext6 \
        python3.11 python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Install pip and set python3 to 3.11
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY MuseTalk/ MuseTalk/
COPY scripts/ scripts/
COPY wheels/ wheels/

# Install lightweight requirements for downloading
RUN pip install --no-cache-dir \
    requests tqdm boto3 runpod concurrent-log-handler

# Install PyTorch + CUDA
RUN pip install --no-cache-dir wheels/*.whl

# Install main project requirements
RUN pip install --no-cache-dir -r MuseTalk/requirements.txt

# Download model weights in parallel
RUN python3 scripts/download_all_weights.py && \
    find /root/.cache -type f -delete

# Final cleanup: remove pip & temporary caches
RUN rm -rf /root/.cache /tmp/* /var/lib/apt/lists/* ~/.cache/pip

# Default entrypoint
CMD ["python3", "scripts/runpod_handler.py"]