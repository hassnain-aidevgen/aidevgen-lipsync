import boto3
import tempfile
import os

# Initialize S3 client only once
s3 = boto3.client("s3")

def download_from_s3(bucket: str, key: str) -> str:
    """
    Downloads a file from S3 and stores it as a temp file.
    Returns the local temp file path.
    """
    suffix = os.path.splitext(key)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    s3.download_fileobj(bucket, key, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name

def upload_to_s3(local_path: str, bucket: str, key: str) -> None:
    """
    Uploads a local file to S3.
    """
    with open(local_path, "rb") as f:
        s3.upload_fileobj(f, bucket, key)

def cleanup(path: str) -> None:
    """
    Safely deletes a file if it exists.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
