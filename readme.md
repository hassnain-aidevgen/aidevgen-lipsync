Thanks for pointing that out — you're absolutely right. Here's the **final updated `README.md`** with:

1. The **Docker commands** to run for testing, pip downloading `.whl` files.
2. Clarification on `%cd%/wheels` usage for Windows (with Docker volume binding).
3. A clean layout for both Windows and Linux/macOS users.

---

### ✅ Final `README.md`

```markdown
# 🧠 MuseTalk Runtime - Docker Build & Run Guide

This project packages the MuseTalk AI lip-sync system into a reproducible Docker image using CUDA 11.7, PyTorch 2.1.2, and Python 3.11. It avoids live downloads during image build by prefetching `.whl` dependencies.

---

## 📦 Prerequisites

- **Docker Desktop**
- **NVIDIA GPU** with [CUDA driver support](https://developer.nvidia.com/cuda-gpus)
- Optional: Python 3.11 installed (to pre-download `.whl` files)

---

## 📁 Project Structure

```

aidevgen\_lipsync/
├── Dockerfile
├── MuseTalk/
├── scripts/
├── wheels/           <-- Put .whl files here (torch, torchvision, torchaudio)
└── README.md

````

---

## ⚙️ Step-by-Step Setup

### 🔹 Option A: Pre-download `.whl` files (Recommended on slow networks)

Run this in a Linux-based Docker container (so you get the **correct `.whl` format**):

```bash
docker run -it --rm -v %cd%/wheels:/wheels python:3.11 bash    # Windows
# OR for Linux/macOS:
docker run -it --rm -v $(pwd)/wheels:/wheels python:3.11 bash
````

Inside the container, run:

```bash
pip install -U pip
pip download torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -d /wheels --extra-index-url https://download.pytorch.org/whl/cu117
exit
```

Make sure your `wheels/` folder on host is now populated with `.whl` files for Linux.

---

### 🔹 Option B: Manual wheel download (if the above fails)

Visit:

* [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)

Download these Linux `.whl` files:

* `torch-2.1.2+cu117-*.whl`
* `torchvision-0.16.2+cu117-*.whl`
* `torchaudio-2.1.2+cu117-*.whl`

Save them inside `wheels/`.

---

## 🛠️ Build the Docker Image

Run from the root project folder:

```bash
docker build -t musetalk-runtime .
```

This will:

* Install OS dependencies
* Set up Python 3.11
* Install `.whl` files from `wheels/`
* Auto-download MuseTalk weights
* Set default entrypoint to `scripts/runpod_handler.py`

---

## ▶️ Run the Container

### Standard run:

```bash
docker run --rm --gpus all musetalk-runtime
```

### With interactive bash:

```bash
docker run -it --rm --gpus all musetalk-runtime bash
```

### Mount a local folder to access output:

```bash
docker run --rm --gpus all -v ${PWD}/output:/app/output musetalk-runtime
```

---

## 🧪 Manual Testing

You can also run individual scripts manually inside the container:

```bash
python3 scripts/download_all_weights.py
python3 scripts/runpod_handler.py
```

---

## 🧹 Cleanup Tips

* Clear Docker system cache:

  ```bash
  docker system prune -a
  ```

* Remove unused images:

  ```bash
  docker image rm <image_id>
  ```

---

## 📌 Notes

* On **Windows**, make sure Docker uses WSL 2 and GPU support is enabled.
* Avoid using Windows `.whl` files — they won't install in Linux containers.
* If downloading fails, try using `--timeout 1000` or `--retries 10`.

---

## ✅ Summary

| Task                  | Command                                                       |
| --------------------- | ------------------------------------------------------------- |
| Download `.whl` files | `docker run -it --rm -v %cd%/wheels:/wheels python:3.11 bash` |
| Build Docker image    | `docker build -t musetalk-runtime .`                          |
| Run the image         | `docker run --rm --gpus all musetalk-runtime`                 |
| Open container shell  | `docker run -it --rm --gpus all musetalk-runtime bash`        |

---

Happy syncing! 🎬🗣️

```

Would you like me to zip up a sample folder structure with this `README.md` included?
```
