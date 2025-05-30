# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Disable prompts and set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install OS packages and clean up in a single layer
# 1) First apt-get update (retry up to 3×) and install core utilities
RUN set -eux; \
    for i in 1 2 3; do \
      apt-get update --allow-releaseinfo-change && break; \
      echo "apt-get update failed, retrying ($i)..."; sleep 5; \
    done; \
    apt-get install -y --no-install-recommends \
      software-properties-common \
      curl wget git unzip ffmpeg \
      libgl1 libsm6 libxext6

# 2) Add deadsnakes PPA, then update again (retry up to 3×)
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    set -eux; \
    for i in 1 2 3; do \
      apt-get update --allow-releaseinfo-change && break; \
      echo "apt-get update after PPA failed, retrying ($i)..."; sleep 5; \
    done

# 3) Install Python 3.11 & distutils
RUN apt-get install -y --no-install-recommends python3.11 python3.11-distutils

# 4) Bootstrap pip under Python 3.11 and symlink
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# 5) Clean up apt caches
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install minimal dependencies for downloads
RUN pip install --no-cache-dir requests tqdm boto3 runpod concurrent-log-handler && \
    rm -rf /root/.cache/pip

# Create MuseTalk app dir and models subdir
RUN mkdir -p /app/MuseTalk/models

# Copy your download script
COPY scripts/download_all_weights.py scripts/__init__.py /app/scripts/

# Skip hash check in download script
RUN sed -i 's/if not skip_hash_check and sha256_checksum(full_path) != expected_hash:/if False:/' \
      /app/scripts/download_all_weights.py

# Download all model weights (once at build; you can move this to cold-start if desired)
# RUN python3 /app/scripts/download_all_weights.py --skip-hash-check --download-one-by-one && \
#     rm -rf /root/.cache/* /tmp/*

# Copy your MuseTalk application code & requirements
COPY MuseTalk/requirements.txt /app/MuseTalk/
COPY MuseTalk/ /app/MuseTalk/

# Clone the upstream MuseTalk repo (depth=1) **into** /app/MuseTalk/upstream
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk/upstream && \
    # Copy just the musetalk/ package into your app
    cp -R /app/MuseTalk/upstream/musetalk /app/MuseTalk/ && \
    # Clean up
    rm -rf /app/MuseTalk/upstream

# Install PyTorch + GPU stack with retries & timeout, then the rest of your Python reqs
RUN pip install --no-cache-dir --timeout 120 --retries 5 \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
      --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir --timeout 120 --retries 5 \
      -r /app/MuseTalk/requirements.txt && \
    rm -rf /root/.cache/pip

# Install OpenMIM and use it to pull in mmengine, mmcv, mmdet & mmpose
RUN pip install --no-cache-dir -U openmim && \
    python3 -m mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"

# Copy your service scripts
COPY scripts/s3_utils.py scripts/musetalk_wrapper.py scripts/runpod_handler.py /app/scripts/

RUN python3 /app/scripts/download_all_weights.py --skip-hash --single && \
    rm -rf /root/.cache/* /tmp/*

# Ensure MuseTalk module and the newly cloned musetalk package can be imported
ENV PYTHONPATH=/app/MuseTalk:$PYTHONPATH
RUN python3 - <<EOF
from musetalk.utils.blending import get_image
print("✅ musetalk code imported successfully")
EOF

# Final cleanup to slim the image
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Default command to run your handler
CMD ["python3", "/app/scripts/runpod_handler.py"]
