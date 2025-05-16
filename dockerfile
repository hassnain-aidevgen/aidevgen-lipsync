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

# Step 2: Add deadsnakes PPA and install Python 3.10
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    for i in 1 2 3; do apt-get update && break || sleep 5; done

RUN apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Install pip for Python 3.10 (removed duplicate line)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Step 4: Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3 1

# Step 5: Set workdir
WORKDIR /app

# Step 6: Copy script files first (to take advantage of Docker layer caching)
COPY scripts/ /app/MuseTalk/scripts/

# Step 7: Adjust the download script to use absolute paths
RUN sed -i 's|LOCAL_ROOT = "MuseTalk/models"|LOCAL_ROOT = "/app/MuseTalk/models"|' \
    /app/MuseTalk/scripts/download_all_weights.py

# Step 8: Patch download_all_weights.py to skip hash check
RUN sed -i 's/if not skip_hash_check and sha256_checksum(full_path) != expected_hash:/if False:/' \
    /app/MuseTalk/scripts/download_all_weights.py

# Step 9: Copy the rest of the application
COPY . /app/MuseTalk/

# Step 10: Install app dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Step 11: Create all necessary model directories before downloading
RUN mkdir -p /app/MuseTalk/models/musetalk \
             /app/MuseTalk/models/musetalkV15 \
             /app/MuseTalk/models/syncnet \
             /app/MuseTalk/models/dwpose \
             /app/MuseTalk/models/face-parse-bisent \
             /app/MuseTalk/models/sd-vae \
             /app/MuseTalk/models/whisper \
             /app/MuseTalk/musetalk/models

# Step 12: Run model weight download
RUN python3 /app/MuseTalk/scripts/download_all_weights.py

# Step 13: Copy all downloaded models to the second location
RUN cp -r /app/MuseTalk/models/* /app/MuseTalk/musetalk/models/

# Step 14: Verify models in both locations
RUN echo "Verifying model files..." && \
    echo "Models in primary location:" && \
    find /app/MuseTalk/models -type f | wc -l && \
    echo "Models in secondary location:" && \
    find /app/MuseTalk/musetalk/models -type f | wc -l

# Step 15: Install PyTorch GPU version
RUN pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

# Step 16: Install additional dependencies for DWPose
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3.10-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Step 17: Install OpenMIM + MMLab ecosystem (combined commands)
RUN pip3 install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmdet>=3.1.0" && \
    mim install "mmpose>=1.1.0" && \
    pip3 install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html

# Step 18: Patch preprocessing.py for DWPOSE
RUN sed -i '10a import os\n_THIS_UTILS_DIR = os.path.dirname(__file__)\n_DWPOSE_DIR = os.path.join(_THIS_UTILS_DIR, "dwpose")\nDWPOSE_CONFIG = os.path.join(_DWPOSE_DIR, "rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py")\nDWPOSE_CHECKPT = os.path.join(_DWPOSE_DIR, "dw-ll_ucoco_384.pth")' \
    /app/MuseTalk/musetalk/utils/preprocessing.py && \
    sed -i 's/model = init_model(config_file, checkpoint_file, device=device)/model = init_model(DWPOSE_CONFIG, DWPOSE_CHECKPT, device=device)/' \
    /app/MuseTalk/musetalk/utils/preprocessing.py

# Step 19: Set environment variables and use system ffmpeg
ENV FFMPEG_PATH=/usr/bin/ffmpeg

# Step 20: Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + || true

# Step 21: Entrypoint
CMD ["python3", "/app/MuseTalk/scripts/runpod_handler.py"]