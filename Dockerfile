# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Disable prompts and set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install OS packages and clean up in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl wget git unzip ffmpeg \
        libgl1 libsm6 libxext6 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-distutils && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Install minimal dependencies for downloads
RUN pip install --no-cache-dir requests tqdm boto3 runpod concurrent-log-handler && \
    rm -rf /root/.cache/pip

# Create model directories
RUN mkdir -p /app/MuseTalk/models

# Copy scripts with minimal dependencies
COPY scripts/download_all_weights.py scripts/__init__.py /app/scripts/

# Modify download script to skip hash check (saves space)
RUN sed -i 's/if not skip_hash_check and sha256_checksum(full_path) != expected_hash:/if False:/' /app/scripts/download_all_weights.py

# Download models with space management
RUN cd /app && python3 scripts/download_all_weights.py --skip-hash-check --download-one-by-one && \
    rm -rf /root/.cache/* /tmp/*

# Copy requirements file
COPY MuseTalk/requirements.txt /app/MuseTalk/

# Filter requirements AND install in a single step to prevent temp file deletion
RUN grep -v "gradio" /app/MuseTalk/requirements.txt > /app/filtered_requirements.txt && \
    pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir -r /app/filtered_requirements.txt && \
    rm -rf /root/.cache/pip

# Copy remaining files
COPY scripts/s3_utils.py scripts/musetalk_wrapper.py scripts/runpod_handler.py /app/scripts/
COPY MuseTalk/*.py /app/MuseTalk/

# Initialize mime types and final cleanup
RUN python3 -c "import mimetypes; mimetypes.init()" && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -type f -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Default command
CMD ["python3", "/app/scripts/runpod_handler.py"]