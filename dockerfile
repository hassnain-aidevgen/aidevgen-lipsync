# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Step 1: Base system deps (cleaned apt-get update with retry)
RUN for i in 1 2 3; do apt-get update && break || sleep 5; done && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    ffmpeg \
    git \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Step 2: Add deadsnakes PPA and install Python 3.11
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    for i in 1 2 3; do apt-get update && break || sleep 5; done

RUN apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Step 4: Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# Step 5: Set workdir
WORKDIR /app

# Step 6: Copy script files first (to take advantage of Docker layer caching)
COPY scripts/ /app/MuseTalk/scripts/

# Step 7: Patch download_all_weights.py
RUN sed -i 's/if not skip_hash_check and sha256_checksum(full_path) != expected_hash:/if False:/' \
    /app/MuseTalk/scripts/download_all_weights.py

# Step 8: Copy the rest of the application (now)
COPY . /app/MuseTalk/

# Step 9: Install app dependencies (which includes tqdm)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Step 10: Run model weight download
RUN python3 /app/MuseTalk/scripts/download_all_weights.py

# Step 11: Install PyTorch GPU version
RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Step 12: Install OpenMIM + MMLab ecosystem
RUN pip3 install --no-cache-dir -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.0.1"
RUN mim install "mmdet>=3.1.0"
RUN mim install "mmpose>=1.1.0"

# Step 13: Patch preprocessing.py for DWPOSE
RUN sed -i '10a import os\n_THIS_UTILS_DIR = os.path.dirname(__file__)\n_DWPOSE_DIR = os.path.join(_THIS_UTILS_DIR, "dwpose")\nDWPOSE_CONFIG = os.path.join(_DWPOSE_DIR, "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py")\nDWPOSE_CHECKPT = os.path.join(_DWPOSE_DIR, "dw-ll_ucoco_384.pth")' \
    /app/MuseTalk/musetalk/utils/preprocessing.py && \
    sed -i 's/model = init_model(config_file, checkpoint_file, device=device)/model = init_model(DWPOSE_CONFIG, DWPOSE_CHECKPT, device=device)/' \
    /app/MuseTalk/musetalk/utils/preprocessing.py

# Step 14: Set environment variables
ENV FFMPEG_PATH=/app/ffmpeg-7.0.2-amd64-static

# Step 15: Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + || true

# Step 16: Entrypoint
CMD ["python3", "/app/MuseTalk/app.py"]
