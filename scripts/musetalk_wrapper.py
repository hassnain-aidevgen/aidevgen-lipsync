import os
import shutil
import subprocess
import uuid
import mimetypes
from MuseTalk.app import inference
from s3_utils import upload_to_s3

def convert_image_to_video(image_path: str, video_path: str, duration: float = 3.0):
    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",
        "-i", image_path,
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", "scale=512:512",
        video_path
    ]
    subprocess.run(cmd, check=True)
    print(f"[INFO] Created dummy video from image: {video_path}")

def is_image_file(path: str) -> bool:
    mime_type, _ = mimetypes.guess_type(path)
    return mime_type and mime_type.startswith("image")

def generate_video(
    audio_path: str,
    image_path: str,
    output_path: str,
    bbox_shift: float = 0.0,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90
) -> str:
    # Convert image to video if needed
    if is_image_file(image_path):
        tmp_video_path = image_path.replace(".jpg", ".mp4").replace(".png", ".mp4")
        convert_image_to_video(image_path, tmp_video_path)
    else:
        tmp_video_path = image_path  # already a video
        print(f"[INFO] Detected video file: {tmp_video_path}")

    print(f"[INFO] Running MuseTalk inference...")
    result_video, _ = inference(
        audio_path,
        tmp_video_path,
        bbox_shift,
        extra_margin,
        parsing_mode,
        left_cheek_width,
        right_cheek_width
    )

    if not os.path.exists(result_video):
        raise FileNotFoundError(f"[ERROR] Inference output not found at {result_video}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.move(result_video, output_path)
    print(f"[INFO] Output saved to {output_path}")

    # Upload to S3
    bucket = os.getenv("S3_BUCKET")
    unique_id = uuid.uuid4().hex
    s3_key_env = os.getenv("S3_KEY")
    s3_key = s3_key_env or f"outputs/musetalk/output_{unique_id}.mp4"

    if bucket:
        print(f"[INFO] Uploading to S3 bucket {bucket} with key {s3_key}")
        upload_to_s3(output_path, bucket, s3_key)
        print(f"[INFO] Uploaded to s3://{bucket}/{s3_key}")

    return output_path
