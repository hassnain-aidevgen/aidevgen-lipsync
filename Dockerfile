# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Disable prompts and set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install OS packages with integrated cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl wget git unzip ffmpeg \
        libgl1 libsm6 libxext6 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-distutils && \
    # Aggressive cleanup after installation
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    # Install pip and set python3 to 3.11
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# First install lightweight core dependencies only
RUN pip install --no-cache-dir \
    requests tqdm boto3 runpod concurrent-log-handler && \
    rm -rf /root/.cache/pip /tmp/*

# Copy script files needed for downloading models
COPY scripts/__init__.py /app/scripts/__init__.py
COPY scripts/download_all_weights.py /app/scripts/download_all_weights.py
COPY scripts/s3_utils.py /app/scripts/s3_utils.py

# Create model directories
RUN mkdir -p /app/MuseTalk/models

# Run model download with space management (one model at a time)
RUN cd /app && \
    # Download only required models first
    PYTHONUNBUFFERED=1 python3 /app/scripts/download_all_weights.py --skip-hash-check --download-one-by-one && \
    # Clean up all temporary files
    rm -rf /root/.cache/pip /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -type f -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Copy requirements file and install remaining dependencies
COPY MuseTalk/requirements.txt /app/MuseTalk/requirements.txt

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    pillow jinja2 markupsafe numpy networkx sympy fsspec mpmath \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir -r /app/MuseTalk/requirements.txt && \
    # Cleanup pip cache immediately
    rm -rf /root/.cache/pip /tmp/* && \
    # Clean up remaining space
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy all necessary application files
COPY MuseTalk/ /app/MuseTalk/
COPY scripts/musetalk_wrapper.py /app/scripts/musetalk_wrapper.py
COPY scripts/runpod_handler.py /app/scripts/runpod_handler.py

# Set mime types for image file detection (used in musetalk_wrapper.py)
RUN python3 -c "import mimetypes; mimetypes.init()"

# Remove unnecessary files to save more space
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    # Clean up Python bytecode
    find /app -name "*.pyc" -type f -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Default command
CMD ["python3", "/app/scripts/runpod_handler.py"]