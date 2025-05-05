import os
import requests
import hashlib
import logging
import argparse
from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "MuseTalk", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Download targets (URL: (dest, hash placeholder))
FILES = {
    # MuseTalk v1.0
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json": (
        "musetalk/musetalk.json",
        "a" * 64,
    ),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin": (
        "musetalk/pytorch_model.bin",
        "b" * 64,
    ),

    # MuseTalk v1.5
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json": (
        "musetalkV15/musetalk.json",
        "c" * 64,
    ),
    "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth": (
        "musetalkV15/unet.pth",
        "d" * 64,
    ),

    # SyncNet
    "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt": (
        "syncnet/latentsync_syncnet.pt",
        "e" * 64,
    ),

    # DWPose
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth": (
        "dwpose/dw-ll_ucoco_384.pth",
        "f" * 64,
    ),

    # Face-parse (BiSeNet)
    "https://download.pytorch.org/models/resnet18-5c106cde.pth": (
        "face-parse-bisent/resnet18-5c106cde.pth",
        "g" * 64,
    ),
    "https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth": (
        "face-parse-bisent/79999_iter.pth",
        "h" * 64,
    ),

    # SD-VAE
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json": (
        "sd-vae/config.json",
        "i" * 64,
    ),
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin": (
        "sd-vae/diffusion_pytorch_model.bin",
        "j" * 64,
    ),

    # Whisper-tiny
    "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json": (
        "whisper/config.json",
        "k" * 64,
    ),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin": (
        "whisper/pytorch_model.bin",
        "l" * 64,
    ),
    "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json": (
        "whisper/preprocessor_config.json",
        "m" * 64,
    ),
}

# Sort files by estimated size (config files first, then small models, then large models)
SMALL_FILES = [url for url in FILES.keys() if url.endswith('.json')]
LARGE_FILES = [url for url in FILES.keys() 
              if any(ext in url for ext in ['pytorch_model.bin', 'unet.pth', 'diffusion_pytorch_model.bin'])]
MEDIUM_FILES = [url for url in FILES.keys() 
               if url not in SMALL_FILES and url not in LARGE_FILES]

def sha256_checksum(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def download_file(url, path, expected_hash, retries=3, skip_hash_check=False):
    full_path = os.path.join(MODELS_DIR, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        if skip_hash_check or sha256_checksum(full_path) == expected_hash:
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

            if not skip_hash_check and sha256_checksum(full_path) != expected_hash:
                raise ValueError("Hash mismatch")
            
            # Return file size (used for space management)
            file_size = os.path.getsize(full_path)
            logging.info(f"[📁] Saved: {path} ({file_size/1024/1024:.1f} MB)")
            return file_size
        except Exception as e:
            logging.error(f"[❌] Attempt {attempt + 1} for {path}: {e}")
            if attempt < retries - 1:
                sleep(2 ** attempt)
            else:
                logging.critical(f"[💥] Failed to download: {path}")

def free_disk_space():
    """Clean up to free disk space during build"""
    os.system("rm -rf /tmp/* /var/tmp/* 2>/dev/null || true")
    os.system("apt-get clean 2>/dev/null || true")
    os.system("rm -rf /root/.cache/* 2>/dev/null || true")
    
def download_in_groups(file_groups, skip_hash_check=False, max_workers=4):
    """Download files in groups, with space cleanup between groups"""
    total_files = sum(len(group) for group in file_groups)
    completed = 0
    
    for i, group in enumerate(file_groups):
        logging.info(f"[🔄] Processing file group {i+1}/{len(file_groups)} ({len(group)} files)")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_file, url, FILES[url][0], FILES[url][1], 
                               retries=3, skip_hash_check=skip_hash_check): url
                for url in group
            }
            for future in as_completed(futures):
                try:
                    file_size = future.result()
                    completed += 1
                    logging.info(f"[🔢] Progress: {completed}/{total_files} files downloaded")
                except Exception as e:
                    url = futures[future]
                    logging.error(f"[ERROR] {FILES[url][0]} failed: {e}")
        
        # Free space between groups
        free_disk_space()
        logging.info(f"[🧹] Cleaned up temporary files after group {i+1}")

def download_one_by_one(files, skip_hash_check=False):
    """Download files one at a time with cleanup between each download"""
    for i, url in enumerate(files):
        dest, hash_val = FILES[url]
        logging.info(f"[🔄] Downloading file {i+1}/{len(files)}: {dest}")
        try:
            download_file(url, dest, hash_val, retries=3, skip_hash_check=skip_hash_check)
            # Free space after each file
            free_disk_space()
            logging.info(f"[🧹] Cleaned up temporary files after downloading {dest}")
        except Exception as e:
            logging.error(f"[ERROR] Failed to download {dest}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download model weights for MuseTalk")
    parser.add_argument("--skip-hash-check", action="store_true", help="Skip hash verification")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel downloads")
    parser.add_argument("--download-one-by-one", action="store_true", 
                       help="Download files one by one to manage disk space")
    args = parser.parse_args()

    # Ensure model directories exist
    for path in set(os.path.dirname(FILES[url][0]) for url in FILES):
        os.makedirs(os.path.join(MODELS_DIR, path), exist_ok=True)

    if args.download_one_by_one:
        # Process in order: small files, medium files, large files
        logging.info("[🔄] Downloading files one by one to manage disk space")
        logging.info(f"[🔄] Downloading {len(SMALL_FILES)} small files")
        download_one_by_one(SMALL_FILES, args.skip_hash_check)
        
        logging.info(f"[🔄] Downloading {len(MEDIUM_FILES)} medium-sized files")
        download_one_by_one(MEDIUM_FILES, args.skip_hash_check)
        
        logging.info(f"[🔄] Downloading {len(LARGE_FILES)} large files")
        download_one_by_one(LARGE_FILES, args.skip_hash_check)
    else:
        # Group downloads by file size
        download_in_groups([SMALL_FILES, MEDIUM_FILES, LARGE_FILES], 
                          skip_hash_check=args.skip_hash_check,
                          max_workers=args.max_workers)

    print("\n[✅] All model downloads complete.")
    
    # Final cleanup
    free_disk_space()
    print("[🧹] Final cleanup completed")

if __name__ == "__main__":
    main()