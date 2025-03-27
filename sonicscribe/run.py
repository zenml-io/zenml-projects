"""This script transcribes audio files from a local folder to text using Gemini API.

The script uses ZenML to define the pipeline and steps.
"""

from zenml import step, pipeline
from zenml.logger import get_logger
import argparse
import os
from google.cloud import storage
from typing import List


# Set up logging with Rich
logger = get_logger(__name__)

GCP_PROJECT_ID = "zenml-core"
GCP_BUCKET_NAME = "gemini-transcribe-test"
GCP_REGION = "europe-north1"

# List of common audio file extensions
AUDIO_EXTENSIONS = [
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".wma",
]


def is_audio_file(file_path: str) -> bool:
    """Check if a file is an audio file based on its extension."""
    return os.path.splitext(file_path)[1].lower() in AUDIO_EXTENSIONS


@step(enable_cache=False)
def upload_audio_file(folder_path: str) -> List[str]:
    """Upload audio files to GCP bucket.

    Args:
        folder_path: Path to folder containing audio files

    Returns:
        List of full paths to all uploaded audio files
    """
    logger.info(
        f"Uploading audio files from `{folder_path}` to GCP bucket `{GCP_BUCKET_NAME}`..."
    )
    client = storage.Client(project=GCP_PROJECT_ID)

    if not client.bucket(GCP_BUCKET_NAME).exists():
        logger.info(f"Creating new bucket: {GCP_BUCKET_NAME}")
        bucket = client.create_bucket(GCP_BUCKET_NAME, location=GCP_REGION)
        logger.info(f"Bucket {GCP_BUCKET_NAME} created in {bucket.location}")
    else:
        logger.info(f"Using existing bucket: {GCP_BUCKET_NAME}")
        bucket = client.bucket(GCP_BUCKET_NAME)

    uploaded_files = []
    audio_files_found = False

    for file in os.listdir(folder_path):
        if is_audio_file(file):
            audio_files_found = True
            full_path = os.path.join(folder_path, file)
            blob = bucket.blob(file)
            blob.upload_from_filename(full_path)
            logger.info(f"Uploaded {file} to {GCP_BUCKET_NAME}")
            uploaded_files.append(full_path)
        else:
            logger.debug(f"Skipping {file} as it is not an audio file")

    if not audio_files_found:
        logger.warning(f"No audio files found in {folder_path}")

    return uploaded_files


@pipeline
def transcribe(folder_path: str):
    """Transcribe audio files from GCP bucket."""
    upload_audio_file(folder_path=folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    args = parser.parse_args()

    transcribe(args.folder_path)
