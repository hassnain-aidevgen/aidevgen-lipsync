#!/usr/bin/env bash
set -euo pipefail

echo "[🛠️] Starting model weight download..."
cd "$(dirname "$0")"

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"/{musetalk,musetalkV15,syncnet,dwpose,face-parse-bisent,sd-vae,whisper}

download_if_missing() {
    local url="$1"
    local target="$2"

    if [ -f "$target" ]; then
        echo "[✅] Already exists: $target"
        return 0
    fi

    echo "[⬇️ ] Downloading $target ..."
    for i in {1..3}; do
        wget -O "$target" "$url" && break
        echo "[⚠️ ] Retry $i failed for: $url"
        sleep 2
    done

    if [ ! -s "$target" ]; then
        echo "[❌] Failed to download after 3 attempts: $url"
    else
        echo "[✅] Successfully downloaded: $target"
    fi
}

# ─── MuseTalk v14 ─────────────────────────
download_if_missing "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json" "$MODELS_DIR/musetalk/musetalk.json"
download_if_missing "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin" "$MODELS_DIR/musetalk/pytorch_model.bin"

# ─── MuseTalk v15 ─────────────────────────
download_if_missing "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json" "$MODELS_DIR/musetalkV15/musetalk.json"
download_if_missing "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth" "$MODELS_DIR/musetalkV15/unet.pth"

# ─── SyncNet (LatentSync) ─────────────────
download_if_missing "https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt" "$MODELS_DIR/syncnet/latentsync_syncnet.pt"

# ─── DW-Pose ──────────────────────────────
download_if_missing "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth" "$MODELS_DIR/dwpose/dw-ll_ucoco_384.pth"

# ─── Face Parse (BiSeNet) ─────────────────
download_if_missing "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812" "$MODELS_DIR/face-parse-bisent/79999_iter.pth"
download_if_missing "https://download.pytorch.org/models/resnet18-5c106cde.pth" "$MODELS_DIR/face-parse-bisent/resnet18-5c106cde.pth"

# ─── SD VAE (optional) ────────────────────
download_if_missing "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json" "$MODELS_DIR/sd-vae/config.json"
download_if_missing "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin" "$MODELS_DIR/sd-vae/diffusion_pytorch_model.bin"

# ─── Whisper Tiny ─────────────────────────
download_if_missing "https://huggingface.co/openai/whisper-tiny/resolve/main/config.json" "$MODELS_DIR/whisper/config.json"
download_if_missing "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin" "$MODELS_DIR/whisper/pytorch_model.bin"
download_if_missing "https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json" "$MODELS_DIR/whisper/preprocessor_config.json"

# ─── Summary ──────────────────────────────
echo ""
echo "📦 Final download summary:"
find "$MODELS_DIR" -type f -exec ls -lh {} \;
echo ""
echo "[✅] Model download script completed (some models may be missing if failed)."
