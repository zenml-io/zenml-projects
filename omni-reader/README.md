# OmniReader - Multi-model text extraction comparison

OmniReader is a document processing workflow that ingests unstructured documents (PDFs, images, scans) and extracts text using multiple OCR models - specifically Gemma 3 and Mistral AI Pixtral12B. The platform provides side-by-side comparison of extraction results, highlighting differences in accuracy, formatting, and content recognition. This dual-model approach allows users to evaluate OCR performance across different document types, languages, and formatting complexity. OmniReader delivers reproducible, automated, and cloud-agnostic analysis, with comprehensive metrics on extraction quality, processing time, and confidence scores for each model.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and configured for Gemma3
- Mistral API key (set as environment variable `MISTRAL_API_KEY`)
- ZenML installed and configured

### Installation

```bash

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Ensure Ollama is running with Gemma3 model:

```bash
ollama pull gemma3:27b
```

2. Set your Mistral API key:

```bash
export MISTRAL_API_KEY=your_mistral_api_key
```

## 📌 Usage

### Running the OCR Pipeline

The project provides a pipeline for comparing OCR capabilities of Gemma3 and Mistral models:

```bash
# Run with default settings (processes all images in assets directory)
python main.py

# Run with custom prompt
python main.py --custom_prompt "Extract all text from this image and identify any named entities."

# Run with specific image directory
python main.py --image_dir /path/to/images

# Run with ground truth for evaluation
python main.py --ground_truth "Text from image 1" "Text from image 2"

# Run with configuration file
python main.py --config_path config.yaml

# For quicker non-ZenML processing of a single image
python run_compare_ocr.py --image assets/your_image.jpg --model both
```

### Using the Streamlit App

For interactive use, the project includes a Streamlit app:

```bash
streamlit run app.py
```

## 📋 Pipeline Architecture

The OCR comparison pipeline consists of the following components:

### Steps

1. **Gemma3 OCR Step**: Uses Ollama Gemma3 model for OCR
2. **Mistral OCR Step**: Uses Mistral's Pixtral model for OCR
3. **Evaluation Step**: Compares results and calculates metrics

### Metadata Tracking

ZenML's metadata tracking is used throughout the pipeline:

- Processing times and performance metrics
- Extracted text length and entity counts
- Comparison metrics between models (CER, WER)

### Results Visualization

- Pipeline results are available in the ZenML Dashboard
- MLflow integration provides detailed metrics and artifacts

## 🔗 Links

- [ZenML Documentation](https://zenml.io/docs)
- [Mistral AI](https://mistral.ai/)
- [Gemma3 Documentation](https://ai.google.dev/gemma3)
