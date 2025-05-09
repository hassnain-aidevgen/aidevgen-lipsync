import os
import requests
import hashlib
import logging
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Determine paths (works both in build and cold-start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "MuseTalk", "models"))

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Files to download: URL -> (relative_path_under_models, expected_hash)
FILES = {
    # MuseTalk v1.0
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json": (
        "musetalk/musetalk.json", "a" * 64),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin": (
        "musetalk/pytorch_model.bin", "b" * 64),
    # MuseTalk v1.5
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json": (
        "musetalkV15/musetalk.json", "c" * 64),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth": (
        "musetalkV15/unet.pth", "d" * 64),
    # SyncNet
    "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt": (
        "syncnet/latentsync_syncnet.pt", "e" * 64),
    # DWPose
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth": (
        "dwpose/dw-ll_ucoco_384.pth", "f" * 64),
    # Face-parse (BiSeNet)
    "https://download.pytorch.org/models/resnet18-5c106cde.pth": (
        "face-parse-bisent/resnet18-5c106cde.pth", "g" * 64),
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth": (
        "face-parse-bisent/79999_iter.pth", "h" * 64),
    # SD-VAE
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json": (
        "sd-vae/config.json", "i" * 64),
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin": (
        "sd-vae/diffusion_pytorch_model.bin", "j" * 64),
    # Whisper-tiny
    "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json": (
        "whisper/config.json", "k" * 64),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin": (
        "whisper/pytorch_model.bin", "l" * 64),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json": (
        "whisper/preprocessor_config.json", "m" * 64),
}

# Group files by size categories
SMALL_FILES = [u for u in FILES if u.endswith('.json')]
LARGE_FILES = [u for u in FILES if any(ext in u for ext in ['pytorch_model.bin', 'unet.pth', 'diffusion_pytorch_model.bin'])]
MEDIUM_FILES = [u for u in FILES if u not in SMALL_FILES + LARGE_FILES]


def sha256_checksum(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def free_disk_space():
    os.system('rm -rf /tmp/* /var/tmp/* 2>/dev/null || true')
    os.system('rm -rf ~/.cache/pip 2>/dev/null || true')


def download_file(url, rel_path, expected_hash, retries=3, skip_hash=False):
    dest = os.path.join(MODELS_DIR, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Skip if already valid
    if os.path.exists(dest) and (skip_hash or sha256_checksum(dest) == expected_hash):
        logging.info(f"[✅] Exists and valid: {rel_path}")
        return

    for attempt in range(1, retries + 1):
        try:
            logging.info(f"[⬇️ ] Downloading (try {attempt}): {rel_path}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=rel_path) as bar:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            if not skip_hash and sha256_checksum(dest) != expected_hash:
                raise ValueError('Hash mismatch')

            logging.info(f"[📁] Downloaded: {rel_path}")
            return
        except Exception as e:
            logging.warning(f"[⚠️] {rel_path} attempt {attempt} failed: {e}")
            sleep(2 ** attempt)

    logging.error(f"[❌] Failed to download after {retries} attempts: {rel_path}")


def download_all_models(skip_hash=False, download_one_by_one=True, max_workers=4):
    # Ensure directories
    for subdir in set(os.path.dirname(FILES[u][0]) for u in FILES):
        os.makedirs(os.path.join(MODELS_DIR, subdir), exist_ok=True)

    if download_one_by_one:
        logging.info("[🔄] Downloading files one by one")
        for group in (SMALL_FILES, MEDIUM_FILES, LARGE_FILES):
            for url in group:
                download_file(url, FILES[url][0], FILES[url][1], skip_hash=skip_hash)
            free_disk_space()
    else:
        logging.info("[🔄] Downloading files in parallel groups")
        for group in (SMALL_FILES, MEDIUM_FILES, LARGE_FILES):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(download_file, url, FILES[url][0], FILES[url][1], skip_hash) for url in group]
                for f in as_completed(futures):
                    f.result()
            free_disk_space()

    logging.info("[✅] All downloads complete")


# CLI entrypoint
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Download model weights for MuseTalk")
    parser.add_argument('--skip-hash', action='store_true', help='Skip hash verification')
    parser.add_argument('--single', action='store_true', help='Download one by one')
    args = parser.parse_args()

    download_all_models(skip_hash=args.skip_hash, download_one_by_one=args.single)
