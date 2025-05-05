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
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Install lightweight dependencies first
RUN pip install --no-cache-dir \
    requests tqdm boto3 runpod concurrent-log-handler && \
    rm -rf /root/.cache/pip /tmp/*

# Create model directories
RUN mkdir -p /app/MuseTalk/models/musetalk \
             /app/MuseTalk/models/musetalkV15 \
             /app/MuseTalk/models/syncnet \
             /app/MuseTalk/models/dwpose \
             /app/MuseTalk/models/face-parse-bisent \
             /app/MuseTalk/models/sd-vae \
             /app/MuseTalk/models/whisper

# Copy script files needed for downloads
COPY scripts/download_all_weights.py /app/scripts/download_all_weights.py
COPY scripts/__init__.py /app/scripts/__init__.py

# Download small files first - config files
RUN wget -q -O /app/MuseTalk/models/musetalk/musetalk.json \
        https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json && \
    wget -q -O /app/MuseTalk/models/musetalkV15/musetalk.json \
        https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json && \
    wget -q -O /app/MuseTalk/models/sd-vae/config.json \
        https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json && \
    wget -q -O /app/MuseTalk/models/whisper/config.json \
        https://huggingface.co/openai/whisper-tiny/resolve/main/config.json && \
    wget -q -O /app/MuseTalk/models/whisper/preprocessor_config.json \
        https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json && \
    rm -rf /tmp/* /var/tmp/*

# Download first medium-sized model
RUN wget -q --show-progress -O /app/MuseTalk/models/syncnet/latentsync_syncnet.pt \
        https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt && \
    rm -rf /tmp/* /var/tmp/*

# Download second medium-sized model
RUN wget -q --show-progress -O /app/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth \
        https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth && \
    rm -rf /tmp/* /var/tmp/*

# Download face parse models
RUN wget -q --show-progress -O /app/MuseTalk/models/face-parse-bisent/resnet18-5c106cde.pth \
        https://download.pytorch.org/models/resnet18-5c106cde.pth && \
    wget -q --show-progress -O /app/MuseTalk/models/face-parse-bisent/79999_iter.pth \
        https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth && \
    rm -rf /tmp/* /var/tmp/*

# Download whisper model
RUN wget -q --show-progress -O /app/MuseTalk/models/whisper/pytorch_model.bin \
        https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin && \
    rm -rf /tmp/* /var/tmp/*

# Download SD-VAE model
RUN wget -q --show-progress -O /app/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin \
        https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin && \
    rm -rf /tmp/* /var/tmp/*

# Download first large model
RUN wget -q --show-progress -O /app/MuseTalk/models/musetalk/pytorch_model.bin \
        https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin && \
    rm -rf /tmp/* /var/tmp/*

# Download second large model
RUN wget -q --show-progress -O /app/MuseTalk/models/musetalkV15/unet.pth \
        https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth && \
    rm -rf /tmp/* /var/tmp/*

# Copy requirements file
COPY MuseTalk/requirements.txt /app/MuseTalk/requirements.txt

# Filter gradio from requirements
RUN grep -v "gradio" /app/MuseTalk/requirements.txt > /tmp/filtered_requirements.txt

# Install PyTorch and other heavy dependencies
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    rm -rf /root/.cache/pip /tmp/*

# Install remaining dependencies
RUN pip install --no-cache-dir -r /tmp/filtered_requirements.txt && \
    rm -rf /root/.cache/pip /tmp/*

# Copy remaining scripts
COPY scripts/s3_utils.py /app/scripts/s3_utils.py
COPY scripts/musetalk_wrapper.py /app/scripts/musetalk_wrapper.py
COPY scripts/runpod_handler.py /app/scripts/runpod_handler.py

# Copy essential MuseTalk files
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