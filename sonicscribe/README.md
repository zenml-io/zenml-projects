# SonicScribe

A pipeline for transcribing audio files using the Gemini API. Built with ZenML.

## Prerequisites

- Python 3.9+
- Google Cloud Platform account with appropriate permissions
- ZenML installed

## Installation

1. Clone this repository
2. Install dependencies:
```bash
uv pip install -r pyproject.toml
```

## Usage

The pipeline can be run using the following commands:

```bash
# Standard usage
python run.py -s /path/to/audio/files -d destination_subfolder

# Synchronous transcription (process immediately)
python run.py -s /path/to/audio/files -d destination_subfolder --synchronous

# Check results of a batch transcription job
python run.py -b
```

### Arguments

- `-s, --source_folder`: Path to the folder containing audio files to transcribe (required)
- `-d, --destination_subfolder`: Optional subfolder within the GCP bucket to store files
- `--synchronous`: Use synchronous transcription instead of batch processing
- `-b, --batch_results`: Get results for a previously submitted batch transcription job

### Examples

```bash
# Basic usage with batch processing (asynchronous)
python run.py --source_folder ./data --destination_subfolder nixon

# Process files immediately (synchronous)
python run.py -s ./data -d nixon --synchronous

# Check results of a previously submitted batch job
python run.py -b
```

### Transcription Modes

1. **Batch Processing (Default)**: 
   - Files are uploaded and processed asynchronously
   - Results are scheduled to be checked automatically after 10 minutes
   - Results are checked every 30 minutes for up to 2 hours

2. **Synchronous Processing**:
   - Files are processed immediately
   - Results are returned directly when processing completes
   - Suitable for small numbers of files

### Supported Audio Formats

The pipeline supports the following audio file formats:
- MP3 (.mp3)
- WAV (.wav)
- OGG (.ogg)
- FLAC (.flac)
- M4A (.m4a)
- AAC (.aac)
- WMA (.wma)

## Features

- Automatic file upload to Google Cloud Storage
- Duplicate file detection to prevent re-uploads
- Support for organizing files in subfolders
- Built-in logging for tracking pipeline progress
- Batch processing using Vertex AI for handling large volumes
- Pipeline scheduling for checking batch processing results
- HTML report generation for viewing transcription results

## Configuration

The pipeline uses the following GCP configuration:
- Project ID: `zenml-core`
- Bucket Name: `gemini-transcribe-test`
- Region: `europe-north1`
- Model: `gemini-2.0-flash-001`

To use different GCP settings, modify the constants in `run.py`:
```python
GCP_PROJECT_ID = "your-project-id"
GCP_BUCKET_NAME = "your-bucket-name"
GCP_REGION = "your-region"
TRANSCRIPTION_MODEL = "your-model"
```