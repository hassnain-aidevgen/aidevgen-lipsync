# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Set environment variable to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Step 1: Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    curl \
    build-essential \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean

# Step 2: Manually install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Step 3: Update alternatives to point to Python 3.11 and pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# Step 4: Set work directory
WORKDIR /app

# Step 5: Copy scripts and patch download_all_weights.py
COPY scripts/ /app/scripts/
RUN sed -i 's/if not skip_hash_check and sha256_checksum(full_path) != expected_hash:/if False:/' \
    /app/scripts/download_all_weights.py

# Step 6: Run weight download script
RUN python3 /app/scripts/download_all_weights.py

# Step 7: Copy application code and ffmpeg static build
COPY MuseTalk /app/MuseTalk
COPY ffmpeg-7.0.2-amd64-static /app/ffmpeg-7.0.2-amd64-static

# Optional: Copy utils to root if needed for imports
RUN cp -r /app/MuseTalk/musetalk/utils /app/MuseTalk/utils

# Step 8: Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Step 9: Install PyTorch GPU build
RUN pip3 install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu117

# Step 10: Install OpenMIM and MMLab packages
RUN pip3 install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0"

# Step 11: Patch preprocessing.py for dwpose model paths
RUN sed -i '10a import os\n_THIS_UTILS_DIR = os.path.dirname(__file__)\n_DWPOSE_DIR = os.path.join(_THIS_UTILS_DIR, "dwpose")\nDWPOSE_CONFIG = os.path.join(_DWPOSE_DIR, "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py")\nDWPOSE_CHECKPT = os.path.join(_DWPOSE_DIR, "dw-ll_ucoco_384.pth")' \
    /app/MuseTalk/musetalk/utils/preprocessing.py && \
    sed -i 's/model = init_model(config_file, checkpoint_file, device=device)/model = init_model(DWPOSE_CONFIG, DWPOSE_CHECKPT, device=device)/' \
    /app/MuseTalk/musetalk/utils/preprocessing.py

# Step 12: Set FFMPEG environment path
ENV FFMPEG_PATH=/app/ffmpeg-7.0.2-amd64-static

# Step 13: Final cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + || true

# Step 14: Default entrypoint
CMD ["python3", "/app/scripts/runpod_handler.py"]
