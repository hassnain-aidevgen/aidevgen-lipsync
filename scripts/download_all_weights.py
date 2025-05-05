import os
import requests
import hashlib
import logging
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "MuseTalk", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Download targets (URL: (dest, hash placeholder))
FILES = {
    # MuseTalk
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json": ("musetalk/musetalk.json", "a"*64),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin": ("musetalk/pytorch_model.bin", "b"*64),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json": ("musetalkV15/musetalk.json", "c"*64),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth": ("musetalkV15/unet.pth", "d"*64),

    # SyncNet
    "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt": ("syncnet/latentsync_syncnet.pt", "e"*64),

    # DWPose
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth": ("dwpose/dw-ll_ucoco_384.pth", "f"*64),

    # Face Parse
    "https://download.pytorch.org/models/resnet18-5c106cde.pth": ("face-parse-bisent/resnet18-5c106cde.pth", "g"*64),
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth": ("face-parse-bisent/79999_iter.pth", "h"*64),

    # SD-VAE
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json": ("sd-vae/config.json", "i"*64),
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin": ("sd-vae/diffusion_pytorch_model.bin", "j"*64),

    # Whisper
    "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json": ("whisper/config.json", "k"*64),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin": ("whisper/pytorch_model.bin", "l"*64),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json": ("whisper/preprocessor_config.json", "m"*64),
}

def sha256_checksum(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def download_file(url, path, expected_hash, retries=3):
    full_path = os.path.join(MODELS_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        if sha256_checksum(full_path) == expected_hash:
            logging.info(f"[✅] Valid: {path}")
            return
        else:
            logging.warning(f"[⚠️] Hash mismatch: {path}, redownloading...")

    for attempt in range(retries):
        try:
            logging.info(f"[⬇️ ] Downloading: {path}")
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with open(full_path, "wb") as f, tqdm(
                    desc=path, total=total or None, unit="B", unit_scale=True
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            if sha256_checksum(full_path) != expected_hash:
                raise ValueError("Hash mismatch")
            logging.info(f"[📁] Saved: {path}")
            return
        except Exception as e:
            logging.error(f"[❌] Attempt {attempt + 1} for {path}: {e}")
            if attempt < retries - 1:
                sleep(2 ** attempt)
            else:
                logging.critical(f"[💥] Failed to download: {path}")

def main(max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, url, dest, hash_val): dest
            for url, (dest, hash_val) in FILES.items()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"[ERROR] {futures[future]} failed: {e}")
    print("\n[✅] All model downloads complete.")

if __name__ == "__main__":
    main()
