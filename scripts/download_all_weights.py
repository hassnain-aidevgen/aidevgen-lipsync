import os
import hashlib
import logging
import requests
from tqdm import tqdm

# --- Settings ---
MAX_RETRIES = 3
SKIP_HASH_CHECK = True  # Change to False to enforce SHA256
LOCAL_ROOT = "MuseTalk/models"

# --- Files to download: URL -> (relative local path, expected SHA256) ---
FILES = {
    # MuseTalk v1.0
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json":
        ("musetalk/musetalk.json", "a" * 64),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin":
        ("musetalk/pytorch_model.bin", "b" * 64),
    
    # MuseTalk v1.5
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json":
        ("musetalkV15/musetalk.json", "c" * 64),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth":
        ("musetalkV15/unet.pth", "d" * 64),
    
    # SyncNet
    "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt":
        ("syncnet/latentsync_syncnet.pt", "e" * 64),
    
    # DWPose
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth":
        ("dwpose/dw-ll_ucoco_384.pth", "f" * 64),
    
    # Face-parse (BiSeNet)
    "https://download.pytorch.org/models/resnet18-5c106cde.pth":
        ("face-parse-bisent/resnet18-5c106cde.pth", "g" * 64),
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth":
        ("face-parse-bisent/79999_iter.pth", "h" * 64),
    
    # SD-VAE
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json":
        ("sd-vae/config.json", "i" * 64),
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin":
        ("sd-vae/diffusion_pytorch_model.bin", "j" * 64),
    
    # Whisper-tiny
    "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json":
        ("whisper/config.json", "k" * 64),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin":
        ("whisper/pytorch_model.bin", "l" * 64),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json":
        ("whisper/preprocessor_config.json", "m" * 64),
}

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# --- Helpers ---
def sha256_checksum(file_path):
    hash_func = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_file(url, rel_path, expected_hash):
    full_path = os.path.join(LOCAL_ROOT, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        if SKIP_HASH_CHECK:
            logging.info(f"[✔️] Skipped existing: {rel_path}")
            return
        elif sha256_checksum(full_path) == expected_hash:
            logging.info(f"[✔️] Already downloaded: {rel_path}")
            return
        else:
            logging.warning(f"[!] Hash mismatch. Re-downloading: {rel_path}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"[↓] Downloading ({attempt}/{MAX_RETRIES}): {rel_path}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(full_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=8192), desc=rel_path, unit="KB", leave=False):
                    if chunk:
                        f.write(chunk)

            if SKIP_HASH_CHECK or sha256_checksum(full_path) == expected_hash:
                logging.info(f"[✔️] Success: {rel_path}")
                return
            else:
                logging.warning(f"[!] Hash mismatch after download: {rel_path}")
        except Exception as e:
            logging.warning(f"[!] Attempt {attempt} failed for {rel_path}: {e}")

    logging.error(f"[❌] Failed to download after {MAX_RETRIES} attempts: {rel_path}")


# --- Main Entry ---
if __name__ == "__main__":
    logging.info("🚀 Starting model asset downloads...")

    for url, (rel_path, expected_hash) in FILES.items():
        download_file(url, rel_path, expected_hash)

    logging.info("✅ All downloads complete.")
