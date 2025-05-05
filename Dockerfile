# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Disable prompts and set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install OS packages and cleanup in the same layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl wget git unzip ffmpeg \
        libgl1 libsm6 libxext6 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3.11-distutils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Install pip and set python3 to 3.11
    curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements file first (for better layer caching)
COPY MuseTalk/requirements.txt /app/MuseTalk/requirements.txt
COPY scripts/download_all_weights.py /app/scripts/download_all_weights.py

# Install dependencies directly in one layer
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    requests tqdm boto3 runpod concurrent-log-handler filelock certifi \
    charset_normalizer idna urllib3 typing_extensions \
    pillow jinja2 markupsafe numpy networkx sympy fsspec mpmath \
    --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir -r /app/MuseTalk/requirements.txt

# Copy all application files
COPY MuseTalk/ /app/MuseTalk/
COPY scripts/ /app/scripts/
COPY entrypoint.sh /app/entrypoint.sh

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Download model weights
RUN python3 /app/scripts/download_all_weights.py && \
    # Execute aggressive pruning to free up space
    # Clean pip cache
    rm -rf /root/.cache/pip && \
    # Remove apt cache and lists
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Clean unused packages
    apt-get autoremove -y && \
    # Remove all Python bytecode files
    find /app -name "*.pyc" -type f -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} +  2>/dev/null || true && \
    # Remove temporary files
    rm -rf /tmp/* /var/tmp/* && \
    # Remove model cache files that aren't needed after download
    find /root/.cache -type f -delete && \
    # Remove any git directories
    find /app -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    # Remove wheels directory if it exists but is empty
    rm -rf /app/wheels

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser && \
    # Make sure user has access to application files
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add a health check (adjust URL path as needed for your application)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default entrypoint
CMD ["python3", "/app/scripts/runpod_handler.py"]