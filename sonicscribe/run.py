"""This script transcribes audio files from a local folder to text using Gemini API.

The script uses ZenML to define the pipeline and steps.
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import vertexai
from google.cloud import storage
from litellm import completion
from pydantic import BaseModel
from vertexai.batch_prediction import BatchPredictionJob
from zenml import pipeline, step
from zenml.client import Client as zenml_client
from zenml.config.schedule import Schedule
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

# Add this constant near the top of the file with other constants
MIME_TYPE_MAPPING = {
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/mp4",
}

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


TRANSCRIPTION_PROMPT = """
Please transcribe this interview, in the format of timecode, speaker, caption.
Use speaker A, speaker B, etc. to identify speakers (unless they are referred to
by name).

You should return the transcript_text, audio_content_type, language, and
content_topic. When you're returning the transcript_text, don't include any extra
text or formatting. Just the raw text of the transcript.
"""


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


def get_html_string(
    transcription_results: Dict[str, Union[Dict, TranscriptionResult]],
) -> HTMLString:
    """Get an HTML string from a dictionary of transcription results.

    Args:
        transcription_results: Dictionary mapping file paths to transcription text or TranscriptionResult objects

    Returns:
        HTMLString: Formatted HTML table of transcription results
    """
    transcription_results_pydantic = []

    for result in transcription_results.values():
        if isinstance(result, TranscriptionResult):
            # If it's already a TranscriptionResult object, use it directly
            transcription_results_pydantic.append(result)
        else:
            # If it's a dictionary, convert it to a TranscriptionResult object
            transcription_results_pydantic.append(
                TranscriptionResult(**result)
            )

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


def get_mime_type(file_uri: str) -> str:
    """Determine the MIME type of a file based on its extension.

    Args:
        file_uri: The URI of the file

    Returns:
        The MIME type of the file

    Raises:
        ValueError: If the file extension is not supported
    """
    file_ext = os.path.splitext(file_uri)[1].lower()
    if file_ext not in MIME_TYPE_MAPPING:
        raise ValueError(f"Unsupported file extension: {file_ext}")
    return MIME_TYPE_MAPPING[file_ext]


def format_for_batch_submission(gcs_uris: List[str], prompt: str) -> List[str]:
    """Format GCS URIs into JSONL format for batch submission to Gemini API.

    Args:
        gcs_uris: List of GCS URIs to audio files
        prompt: The transcription prompt to use

    Returns:
        List of JSONL strings ready for batch submission

    Raises:
        ValueError: If any of the file extensions are not supported
    """
    jsonl_strings = []

    for uri in gcs_uris:
        try:
            mime_type = get_mime_type(uri)

            # Create the JSONL entry
            request = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt},
                                {
                                    "fileData": {
                                        "fileUri": uri,
                                        "mimeType": mime_type,
                                    }
                                },
                            ],
                        }
                    ]
                }
            }

            jsonl_strings.append(json.dumps(request))
        except ValueError as e:
            logger.error(f"Error processing file {uri}: {str(e)}")
            raise

    return jsonl_strings


def batch_transcribe_audio_files(gcs_uris: List[str]) -> BatchPredictionJob:
    """Submits a batch transcription request to Gemini API.

    Args:
        gcs_uris: List of GCS URIs to audio files

    Returns:
        None: Results will be available in GCS after job completion
    """
    logger.info(f"Starting batch transcription for {len(gcs_uris)} files")

    # Initialize VertexAI with the project
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)

    # Create a temporary GCS location for the JSONL input file
    jsonl_strings = format_for_batch_submission(gcs_uris, TRANSCRIPTION_PROMPT)
    jsonl_content = "\n".join(jsonl_strings)

    # Create a blob for the input JSONL file
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCP_BUCKET_NAME)
    timestamp = int(time.time())
    input_blob_name = f"batch_inputs/input-{timestamp}.jsonl"
    input_blob = bucket.blob(input_blob_name)

    # Upload the JSONL content
    input_blob.upload_from_string(jsonl_content)
    logger.info(
        f"Uploaded batch input file to gs://{GCP_BUCKET_NAME}/{input_blob_name}"
    )

    # Define input and output URIs
    input_uri = f"gs://{GCP_BUCKET_NAME}/{input_blob_name}"
    output_uri_prefix = (
        f"gs://{GCP_BUCKET_NAME}/batch_outputs/output-{timestamp}"
    )

    # Submit a batch prediction job with Gemini model
    batch_prediction_job = BatchPredictionJob.submit(
        source_model=TRANSCRIPTION_MODEL,
        input_dataset=input_uri,
        output_uri_prefix=output_uri_prefix,
    )

    logger.info(
        f"Batch job submitted with resource name: {batch_prediction_job.resource_name}"
    )
    logger.info(
        f"Model resource name with the job: {batch_prediction_job.model_name}"
    )
    logger.info(f"Job state: {batch_prediction_job.state.name}")
    logger.info(f"Results will be available at: {output_uri_prefix}")

    return batch_prediction_job


@step
def transcribe_audio_file(
    gcs_uri_json_string: str,
    synchronous: bool = False,
) -> Annotated[
    Union[Tuple[Dict[str, str], HTMLString], str],
    "transcription_output",
]:
    """Transcribe audio file using Gemini API.

    Args:
        gcs_uri_json_string: JSON string list of GCS URIs
        synchronous: Whether to use synchronous transcription

    Returns:
        Tuple containing transcription results dictionary and HTML string
        Note: When using batch transcription, results dictionary will be empty
    """
    logger.info(
        f"{'Running synchronous transcription' if synchronous else 'Scheduling batch transcription'}"
    )
    gcs_uris = json.loads(gcs_uri_json_string)

    transcription_results = {}
    if synchronous:
        for uri in gcs_uris:
            transcription_results[uri] = synchronous_transcribe_file(uri)
        return (
            transcription_results,
            get_html_string(transcription_results),
        )
    else:
        batch_prediction_job = batch_transcribe_audio_files(gcs_uris)
        logger.info(
            "Batch transcription job submitted. Results will be available in GCS when completed."
        )
        logger.info(
            "No results returned to pipeline as batch processing happens asynchronously."
        )
        return batch_prediction_job.name


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
    if not synchronous:
        batch_transcription_job_name = transcribe_audio_file(
            gcs_uri_json_string=gcs_uris, synchronous=synchronous
        )
        batch_transcription_results.with_options(
            schedule=Schedule(
                start_time=datetime.now() + timedelta(minutes=10),
                interval_second=1800,  # 30 minutes in seconds
                end_time=datetime.now() + timedelta(hours=2),
            )
        )(batch_transcription_job_name)
    else:
        transcribe_audio_file(
            gcs_uri_json_string=gcs_uris, synchronous=synchronous
        )


def get_batch_prediction_job(
    batch_prediction_job_name: str,
) -> BatchPredictionJob:
    """Get a batch prediction job by name."""
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    job_state_filter = 'state="JOB_STATE_SUCCEEDED"'
    batch_jobs = BatchPredictionJob.list(filter=job_state_filter)

    if transcription_job := [
        job for job in batch_jobs if job.name == batch_prediction_job_name
    ]:
        return transcription_job[0]
    else:
        raise ValueError(
            f"Batch prediction job {batch_prediction_job_name} not found"
        )


@step
def get_batch_transcription_results(
    batch_prediction_job_name: str,
) -> Tuple[
    Annotated[Dict[str, TranscriptionResult], "transcription_results"],
    Annotated[HTMLString, "html_representation"],
]:
    """Get the results of a batch transcription job.

    Args:
        batch_prediction_job_name: The name of the batch transcription job

    Returns:
        Tuple containing transcription results dictionary and HTML string representation
    """
    try:
        batch_prediction_job = get_batch_prediction_job(
            batch_prediction_job_name
        )
    except ValueError:
        logger.info(
            f"Batch prediction job `{batch_prediction_job_name}` has not yet completed. Please check back later."
        )
        return {}, HTMLString(
            f"""<!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #555; text-align: center; }}
                    .message {{ padding: 15px; background-color: #f8f9fa; border-left: 5px solid #6c757d; }}
                </style>
            </head>
            <body>
                <h1>Job Status</h1>
                <div class="message">
                    <p>Batch prediction job <code>{batch_prediction_job_name}</code> has not yet completed.</p>
                    <p>Please check back later.</p>
                </div>
            </body>
            </html>"""
        )

    if batch_prediction_job.has_succeeded:
        logger.info("Job succeeded!")
    else:
        logger.error(f"Job failed: {batch_prediction_job.error}")
        return {}, HTMLString("<h1>Batch prediction job failed</h1>")

    # Check the location of the output
    logger.info(f"Job output location: {batch_prediction_job.output_location}")

    # Download the output from the GCS bucket
    predictions_file_gcs_path = batch_prediction_job.output_location
    # Extract just the path part without the bucket name and gs:// prefix
    parsed_uri = urlparse(predictions_file_gcs_path)
    blob_prefix = parsed_uri.path.lstrip("/")

    # List blobs to find the predictions file
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket_name = parsed_uri.netloc
    bucket = client.bucket(bucket_name)

    # Look for predictions.jsonl in the output location
    predictions_blob_name = None
    for blob in bucket.list_blobs(prefix=blob_prefix):
        if blob.name.endswith("predictions.jsonl"):
            predictions_blob_name = blob.name
            break

    if not predictions_blob_name:
        logger.error(
            f"Could not find predictions.jsonl in {predictions_file_gcs_path}"
        )
        return {}, HTMLString("<h1>Could not find predictions file</h1>")

    local_predictions_file = "predictions.jsonl"
    output_blob = bucket.blob(predictions_blob_name)
    output_blob.download_to_filename(local_predictions_file)

    transcription_results = {}
    try:
        with open(local_predictions_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                result_json = json.loads(line)

                # Extract file URI from the request
                request_content = result_json.get("request", {}).get(
                    "contents", [{}]
                )[0]
                file_data = None
                for part in request_content.get("parts", []):
                    if part.get("fileData"):
                        file_data = part.get("fileData")
                        break

                if not file_data or not file_data.get("fileUri"):
                    logger.warning(
                        f"Could not find file URI in request: {result_json}"
                    )
                    continue

                file_uri = file_data.get("fileUri")
                file_name = file_uri.split("/")[-1]

                # Extract transcript from the response
                response = result_json.get("response", {})
                candidates = response.get("candidates", [])
                if not candidates:
                    logger.warning(f"No candidates found for {file_name}")
                    continue

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    logger.warning(f"No content parts found for {file_name}")
                    continue

                transcript_text = parts[0].get("text", "")

                # Create a TranscriptionResult object
                try:
                    # Try to extract structured data from transcript if possible
                    # This is highly dependent on your model response format
                    # For now, we'll use placeholder values for the structured fields
                    transcription_result = TranscriptionResult(
                        transcript_text=transcript_text,
                        audio_content_type=AudioContentType.INTERVIEW,  # Assuming default
                        language="en",  # Assuming English
                        content_topic=ContentTopic.OTHER,  # Default topic
                    )
                    transcription_results[file_name] = transcription_result
                except Exception as e:
                    logger.error(
                        f"Error creating TranscriptionResult for {file_name}: {str(e)}"
                    )
                    # Fall back to a basic structure if parsing fails
                    transcription_results[file_name] = {
                        "transcript_text": transcript_text,
                        "audio_content_type": AudioContentType.OTHER.value,
                        "language": "en",
                        "content_topic": ContentTopic.OTHER.value,
                    }
    except Exception as e:
        logger.error(f"Error processing JSONL file: {str(e)}")
        os.remove(local_predictions_file)
        return {}, HTMLString(f"<h1>Error processing results: {str(e)}</h1>")

    # Delete the output file
    os.remove(local_predictions_file)

    # Generate HTML from the parsed results
    html_representation = get_html_string(transcription_results)

    return transcription_results, html_representation


@pipeline(enable_cache=False)
def batch_transcription_results(batch_prediction_job_name: str):
    """Pipeline to get batch transcription results.

    Args:
        batch_prediction_job_name: The name of the batch prediction job
    """
    get_batch_transcription_results(batch_prediction_job_name)


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
    parser.add_argument(
        "-b",
        "--batch_results",
        action="store_true",
        help="Get batch transcription results manually (default: False)",
    )
    args = parser.parse_args()

    if args.batch_results:
        batch_prediction_job_name = (
            zenml_client().get_artifact_version("transcription_output").load()
        )
        batch_transcription_results(batch_prediction_job_name)
    else:
        audio_transcription(
            args.source_folder, args.destination_subfolder, args.synchronous
        )
