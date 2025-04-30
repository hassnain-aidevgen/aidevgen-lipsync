@echo off
setlocal enabledelayedexpansion

REM create model dirs
md models\musetalk     2>nul
md models\musetalkV15  2>nul
md models\syncnet      2>nul
md models\dwpose       2>nul
md models\face-parse-bisent 2>nul
md models\sd-vae       2>nul
md models\whisper      2>nul

echo Downloading MuseTalk v14...
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/musetalk.json -OutFile models\musetalk\musetalk.json"
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk/pytorch_model.bin -OutFile models\musetalk\pytorch_model.bin"

echo Downloading MuseTalk v15...
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json -OutFile models\musetalkV15\musetalk.json"
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/unet.pth -OutFile models\musetalkV15\unet.pth"

echo Downloading SyncNet (LatentSync)...
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/ByteDance/LatentSync/resolve/main/latentsync_syncnet.pt -OutFile models\syncnet\latentsync_syncnet.pt"

echo Downloading DW-Pose...
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth -OutFile models\dwpose\dw-ll_ucoco_384.pth"

echo Downloading Face-Parse (BiSeNet)...
powershell -Command "Invoke-WebRequest -Uri https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812 -OutFile models\face-parse-bisent\79999_iter.pth"
powershell -Command "Invoke-WebRequest -Uri https://download.pytorch.org/models/resnet18-5c106cde.pth -OutFile models\face-parse-bisent\resnet18-5c106cde.pth"

echo Downloading SD VAE...
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json -OutFile models\sd-vae\config.json"
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin -OutFile models\sd-vae\diffusion_pytorch_model.bin"

echo Downloading Whisper Tiny...
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/openai/whisper-tiny/resolve/main/config.json -OutFile models\whisper\config.json"
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin -OutFile models\whisper\pytorch_model.bin"
powershell -Command "Invoke-WebRequest -Uri https://huggingface.co/openai/whisper-tiny/resolve/main/preprocessor_config.json -OutFile models\whisper\preprocessor_config.json"

echo Models downloaded successfully!
endlocal
