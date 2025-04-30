import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "MuseTalk", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"[✅] Exists: {path}")
        return
    print(f"[⬇️ ] Downloading: {path}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[📁] Saved to: {path}")
    else:
        print(f"[❌] Failed: {url} — HTTP {r.status_code}")

# MuseTalk
download("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json", f"{MODELS_DIR}/musetalk/musetalk.json")
download("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin", f"{MODELS_DIR}/musetalk/pytorch_model.bin")
download("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json", f"{MODELS_DIR}/musetalkV15/musetalk.json")
download("https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth", f"{MODELS_DIR}/musetalkV15/unet.pth")

# SyncNet
download("https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt", f"{MODELS_DIR}/syncnet/latentsync_syncnet.pt")

# DWPose
download("https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth", f"{MODELS_DIR}/dwpose/dw-ll_ucoco_384.pth")

# Face Parse
download("https://download.pytorch.org/models/resnet18-5c106cde.pth", f"{MODELS_DIR}/face-parse-bisent/resnet18-5c106cde.pth")
download("https://huggingface.co/camenduru/MuseTalk/resolve/main/face-parse-bisent/79999_iter.pth", f"{MODELS_DIR}/face-parse-bisent/79999_iter.pth")

# SD-VAE
download("https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json", f"{MODELS_DIR}/sd-vae/config.json")
download("https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin", f"{MODELS_DIR}/sd-vae/diffusion_pytorch_model.bin")

# Whisper
download("https://huggingface.co/openai/whisper-tiny/resolve/main/config.json", f"{MODELS_DIR}/whisper/config.json")
download("https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin", f"{MODELS_DIR}/whisper/pytorch_model.bin")
download("https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json", f"{MODELS_DIR}/whisper/preprocessor_config.json")

print("\n[✅] All model downloads complete.")
