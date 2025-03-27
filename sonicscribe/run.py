"""This script transcribes audio files from a local folder to text using Gemini API.

The script uses ZenML to define the pipeline and steps.
"""

import argparse
import json
import os
from typing import List, Optional

from google.cloud import storage
from zenml import pipeline, step
from zenml.logger import get_logger

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


def get_gcs_uri(
    bucket_name: str, blob_name: str, subfolder: Optional[str] = None
) -> str:
    """Get the GCS URI for a given bucket and blob name.

    Args:
        bucket_name: Name of the GCP bucket
        blob_name: Name of the blob (file)
        subfolder: Optional subfolder within the bucket

    Returns:
        GCS URI in the format gs://bucket/subfolder/filename or gs://bucket/filename
    """
    if subfolder:
        return f"gs://{bucket_name}/{subfolder}/{blob_name}"
    return f"gs://{bucket_name}/{blob_name}"


def get_json_string_list(gcs_uris: List[str]) -> str:
    """Get a JSON string list of GCS URIs."""
    return json.dumps(gcs_uris)


@step(enable_cache=False)
def upload_audio_file(
    folder_path: str, subfolder: Optional[str] = None
) -> str:
    """Upload audio files to GCP bucket.

    Args:
        folder_path: Path to folder containing audio files
        subfolder: Optional subfolder within the bucket to store files

    Returns:
        JSON string list of GCS URIs to all uploaded audio files
    """
    logger.info(
        f"Uploading audio files from `{folder_path}` to GCP bucket `{GCP_BUCKET_NAME}`"
        + (f" in subfolder `{subfolder}`" if subfolder else "")
        + "..."
    )
    client = storage.Client(project=GCP_PROJECT_ID)

    if not client.bucket(GCP_BUCKET_NAME).exists():
        logger.info(f"Creating new bucket: `{GCP_BUCKET_NAME}`")
        bucket = client.create_bucket(GCP_BUCKET_NAME, location=GCP_REGION)
        logger.info(
            f"Bucket `{GCP_BUCKET_NAME}` created in `{bucket.location}`"
        )
    else:
        logger.info(f"Using existing bucket: `{GCP_BUCKET_NAME}`")
        bucket = client.bucket(GCP_BUCKET_NAME)

    uploaded_files = []
    audio_files_found = False

    for file in os.listdir(folder_path):
        if is_audio_file(file):
            audio_files_found = True
            full_path = os.path.join(folder_path, file)

            # Construct the blob path including subfolder if specified
            blob_path = f"{subfolder}/{file}" if subfolder else file
            blob = bucket.blob(blob_path)

            # Check if file already exists
            if blob.exists():
                logger.info(
                    f"File `{blob_path}` already exists in bucket, skipping upload"
                )
                uploaded_files.append(
                    get_gcs_uri(GCP_BUCKET_NAME, file, subfolder)
                )
                continue

            blob.upload_from_filename(full_path)
            logger.info(
                f"Uploaded `{file}` to `{GCP_BUCKET_NAME}/{blob_path}`"
            )
            uploaded_files.append(
                get_gcs_uri(GCP_BUCKET_NAME, file, subfolder)
            )
        else:
            logger.debug(f"Skipping {file} as it is not an audio file")

    if not audio_files_found:
        logger.warning(f"No audio files found in {folder_path}")

    return get_json_string_list(uploaded_files)


@step
def transcribe_audio_file(gcs_uri_json_string: str) -> str:
    """Transcribe audio file using Gemini API."""
    return "transcribed text"


@pipeline
def audio_transcription(folder_path: str, subfolder: Optional[str] = None):
    """Transcribe audio files from GCP bucket.

    Args:
        folder_path: Path to folder containing audio files
        subfolder: Optional subfolder within the bucket to store files
    """
    gcs_uris = upload_audio_file(folder_path=folder_path, subfolder=subfolder)
    transcribe_audio_file(gcs_uri_json_string=gcs_uris)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source_folder",
        type=str,
        required=True,
        help="Path to folder containing audio files to transcribe",
    )
    parser.add_argument(
        "-d",
        "--destination_subfolder",
        type=str,
        help="Optional subfolder within the bucket to store files",
    )
    args = parser.parse_args()

    audio_transcription(args.source_folder, args.destination_subfolder)
