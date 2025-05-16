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
# from download_all_weights import download_all_models  # Ensure you have this function

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Download models on cold start
# download_all_models()

def download_from_url(url, suffix):
    """Download file from URL and save to a temporary file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure we raise an error for bad responses
        with open(tmp.name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise
    return tmp.name

def handler(event):
    input_data = event.get("input", {})
    bucket = input_data.get("bucket")
    audio_url = input_data.get("audio_url")
    video_url = input_data.get("video_url")

    # Ensure all inputs are provided
    if not all([bucket, audio_url, video_url]):
        return {"status": "error", "message": "Missing bucket/audio_url/video_url"}

    try:
        logger.info(f"Downloading audio from {audio_url} and video from {video_url}")
        
        # Download audio and video
        audio_path = download_from_url(audio_url, ".wav")
        video_suffix = os.path.splitext(video_url)[-1] or ".mp4"
        video_path = download_from_url(video_url, video_suffix)

        # Generate output file name and path
        output_name = f"output_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join("/tmp", output_name)

        # Run the video generation (inference step)
        logger.info("Starting video generation...")
        generate_video(audio_path, video_path, output_path)

        # Upload the result to S3
        output_key = f"outputs/{output_name}"
        upload_to_s3(output_path, bucket, output_key)

        logger.info(f"Video generated and uploaded successfully: {output_key}")
        return {"status": "completed", "output_key": output_key}

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

    finally:
        # Cleanup temporary files
        temp_files = [audio_path, video_path, output_path]
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                cleanup(temp_file)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
