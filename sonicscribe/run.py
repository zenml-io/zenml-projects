"""This script transcribes audio files from a local folder to text using Gemini API.

The script uses ZenML to define the pipeline and steps.
"""

import argparse
import json
import os
from enum import Enum
from typing import Annotated, Dict, List, Optional, Tuple

from google.cloud import storage
from litellm import completion
from pydantic import BaseModel
from rich import print
from zenml import pipeline, step
from zenml.logger import get_logger
from zenml.types import HTMLString

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

TRANSCRIPTION_PROMPT = """
Can you transcribe this interview, in the format of timecode, speaker, caption.
Use speaker A, speaker B, etc. to identify speakers (unless they are referred to by name).
"""

TRANSCRIPTION_MODEL = "gemini-2.0-flash-001"


class AudioContentType(str, Enum):
    INTERVIEW = "interview"
    PHONE_CALL = "phone_call"
    SPEECH = "speech"
    MEETING = "meeting"
    OTHER = "other"


class ContentTopic(str, Enum):
    POLITICAL = "political"
    BUSINESS = "business"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    OTHER = "other"


class TranscriptionResult(BaseModel):
    transcript_text: str
    audio_content_type: AudioContentType
    language: str
    content_topic: ContentTopic


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


@step
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


def get_html_string(transcription_results: Dict[str, str]) -> HTMLString:
    """Get an HTML string from a dictionary of transcription results.

    Args:
        transcription_results: Dictionary mapping file paths to transcription text

    Returns:
        HTMLString: Formatted HTML table of transcription results
    """
    transcription_results_pydantic = [
        TranscriptionResult(**result)
        for result in transcription_results.values()
    ]

    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        table { border-collapse: collapse; width: 100%; }",
        "        th { background-color: #f2f2f2; text-align: left; padding: 12px; }",
        "        td { border: 1px solid #ddd; padding: 12px; vertical-align: top; }",
        "        td.transcription { white-space: pre-wrap; font-family: monospace; }",
        "        tr:nth-child(even) { background-color: #f9f9f9; }",
        "        tr:hover { background-color: #f1f1f1; }",
        "        .metadata { font-size: 0.9em; color: #666; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Audio Transcription Results</h1>",
        "    <table>",
        "        <tr>",
        "            <th>File</th>",
        "            <th>Content Type</th>",
        "            <th>Language</th>",
        "            <th>Topic</th>",
        "            <th>Transcription</th>",
        "        </tr>",
    ]

    # Add each transcription result as a table row
    for file_path, result in zip(
        transcription_results.keys(), transcription_results_pydantic
    ):
        html_parts.append(f"        <tr>")
        html_parts.append(f"            <td>{file_path}</td>")
        html_parts.append(
            f"            <td>{result.audio_content_type.value}</td>"
        )
        html_parts.append(f"            <td>{result.language}</td>")
        html_parts.append(f"            <td>{result.content_topic.value}</td>")
        html_parts.append(
            f'            <td class="transcription">{result.transcript_text}</td>'
        )
        html_parts.append(f"        </tr>")

    # Close the HTML structure
    html_parts.extend(["    </table>", "</body>", "</html>"])

    return HTMLString("\n".join(html_parts))


def synchronous_transcribe_file(
    uri: str,
    model: str = TRANSCRIPTION_MODEL,
) -> str:
    """Transcribe audio file synchronously."""
    logger.info(
        "Sending to Gemini for transcription... (this may take a while)"
    )
    response = completion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": TRANSCRIPTION_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": uri,
                    },
                ],
            }
        ],
        response_format=TranscriptionResult,
    )
    return json.loads(response.choices[0].message.content)


@step
def transcribe_audio_file(
    gcs_uri_json_string: str,
    synchronous: bool = False,
) -> Tuple[
    Annotated[Dict[str, str], "transcription_results"],
    Annotated[HTMLString, "transcription_results_html"],
]:
    """Transcribe audio file using Gemini API."""
    logger.info(
        f"{'Running synchronous transcription' if synchronous else 'Scheduling batch transcription'}"
    )
    # split the json string into a list of gcs uris
    gcs_uris = json.loads(gcs_uri_json_string)

    transcription_results = {}
    if synchronous:
        for uri in gcs_uris:
            transcription_results[uri] = synchronous_transcribe_file(uri)
    else:
        print("doing async")

    return transcription_results, get_html_string(transcription_results)


@pipeline(enable_cache=False)
def audio_transcription(
    folder_path: str,
    subfolder: Optional[str] = None,
    synchronous: bool = False,
):
    """Transcribe audio files from GCP bucket.

    Args:
        folder_path: Path to folder containing audio files
        subfolder: Optional subfolder within the bucket to store files
        synchronous: Whether to use synchronous transcription (default: False)
    """
    gcs_uris = upload_audio_file(folder_path=folder_path, subfolder=subfolder)
    transcribe_audio_file(
        gcs_uri_json_string=gcs_uris, synchronous=synchronous
    )


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
    parser.add_argument(
        "--synchronous",
        action="store_true",
        help="Use synchronous transcription (default: False)",
    )
    args = parser.parse_args()

    audio_transcription(
        args.source_folder, args.destination_subfolder, args.synchronous
    )
