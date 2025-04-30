import os
import sys
import uuid
import logging
import runpod
import tempfile
import requests

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from s3_utils import upload_to_s3, cleanup
from musetalk_wrapper import generate_video

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def download_from_url(url, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(tmp.name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return tmp.name

def handler(event):
    input_data = event.get("input", {})
    bucket = input_data.get("bucket")
    audio_url = input_data.get("audio_url")
    video_url = input_data.get("video_url")

    if not all([bucket, audio_url, video_url]):
        return {"status": "error", "message": "Missing bucket/audio_url/video_url"}

    try:
        logger.info(f"Downloading from {audio_url} and {video_url}")
        audio_path = download_from_url(audio_url, ".wav")
        video_path = download_from_url(video_url, os.path.splitext(video_url)[-1])

        output_name = f"output_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join("/tmp", output_name)

        generate_video(audio_path, video_path, output_path)

        output_key = f"outputs/{output_name}"
        upload_to_s3(output_path, bucket, output_key)

        return {"status": "completed", "output_key": output_key}

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

    finally:
        for path in [audio_path, video_path, output_path]:
            cleanup(path)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
