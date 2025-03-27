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

The pipeline can be run using the following command:

```bash
python run.py -s /path/to/audio/files -d destination_subfolder
```

### Arguments

- `-s, --source_folder`: Path to the folder containing audio files to transcribe (required)
- `-d, --destination_subfolder`: Optional subfolder within the GCP bucket to store files

### Examples

```bash
# Using full argument names
python run.py --source_folder ./data --destination_subfolder nixon

# Using shortened versions
python run.py -s ./data -d nixon

# Mix and match
python run.py -s ./data --destination_subfolder nixon
```

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

## Configuration

The pipeline uses the following GCP configuration:
- Project ID: `zenml-core`
- Bucket Name: `gemini-transcribe-test`
- Region: `europe-north1`

To use different GCP settings, modify the constants in `run.py`:
```python
GCP_PROJECT_ID = "your-project-id"
GCP_BUCKET_NAME = "your-bucket-name"
GCP_REGION = "your-region"
```
