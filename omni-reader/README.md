# OmniReader - Multi-model text extraction comparison

OmniReader is a document processing workflow that ingests unstructured documents (PDFs, images, scans) and extracts text using multiple OCR models - specifically Gemma 3 and Mistral AI Pixtral12B. It provides side-by-side comparison of extraction results, highlighting differences in accuracy, formatting, and content recognition. This dual-model approach allows users to evaluate OCR performance across different document types, languages, and formatting complexity. OmniReader delivers reproducible, automated, and cloud-agnostic analysis, with comprehensive metrics on extraction quality, processing time, and confidence scores for each model.

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

1. Ensure Ollama is running with the required models:

```bash
# For using the default Gemma3 model
ollama pull gemma3:27b

# If using other Ollama models in your config, pull those as well
# ollama pull llama3:70b
# ollama pull dolphin-mixtral:8x7b
```

NOTE: When using Ollama models in your configuration, you must ensure they are pulled locally on your system.

2. Set your Mistral API key:

```bash
export MISTRAL_API_KEY=your_mistral_api_key
```

## 📌 Usage

### Using YAML Configuration (Recommended)

```bash
# Generate a default configuration file
python run.py --create-default-config my_config.yaml

# Run with a configuration file
python run.py --config my_config.yaml

# Generate a configuration for a specific experiment
python config_generator.py --name invoice_test --image-folder ./invoices --ground-truth-source openai
```

### Configuration Structure

YAML configuration files organize parameters into logical sections:

```yaml
# Image input configuration
input:
  image_paths: [] # List of specific image paths
  image_folder: "./assets" # Folder containing images

# OCR model configuration
models:
  custom_prompt: null # Optional custom prompt for both models
  model1: "ollama/gemma3:27b" # First model for comparison
  model2: "pixtral-12b-2409" # Second model for comparison
  ground_truth_model: "gpt-4o-mini" # Model to use for ground truth when source is "openai"

# Ground truth configuration
ground_truth:
  source: "openai" # Source: "openai", "manual", "file", or "none"
  texts: [] # Ground truth texts (for manual source)
  file: null # Path to ground truth JSON file (for file source)

# Output configuration
output:
  ground_truth:
    save: false # Whether to save ground truth data
    directory: "ocr_results" # Directory for ground truth data

  ocr_results:
    save: false # Whether to save OCR results
    directory: "ocr_results" # Directory for OCR results

  visualization:
    save: false # Whether to save HTML visualization
    directory: "visualizations" # Directory for visualization
```

### Running with Command Line Arguments

You can still use command line arguments for quick runs:

```bash
# Run with default settings (processes all images in assets directory)
python run.py --image-folder assets

# Run with custom prompt
python run.py --image-folder assets --custom-prompt "Extract all text from this image."

# Run with specific images
python run.py --image-paths assets/image1.jpg assets/image2.png

# Run with OpenAI ground truth for evaluation
python run.py --image-folder assets --ground-truth openai --save-ground-truth

# List available ground truth files
python run.py --list-ground-truth-files

# For quicker processing of a single image without metadata or artifact tracking
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

1. **Model 1 OCR Step**: Processes images with the first model (default: Ollama Gemma3)
2. **Model 2 OCR Step**: Processes images with the second model (default: Pixtral 12B)
3. **Ground Truth Step**: Optional step that uses a reference model for evaluation (default: GPT-4o Mini)
4. **Evaluation Step**: Compares results and calculates metrics

The pipeline now supports configurable models, allowing you to easily swap out the models used for OCR comparison and ground truth generation via the YAML configuration file.

### Configuration Management

The new configuration system provides:

- Structured YAML files for experiment parameters
- Parameter validation and intelligent defaults
- Easy sharing and version control of experiment settings
- Configuration generator for quickly creating new experiment setups

### Metadata Tracking

ZenML's metadata tracking is used throughout the pipeline:

- Processing times and performance metrics
- Extracted text length and entity counts
- Comparison metrics between models (CER, WER)

### Results Visualization

- Pipeline results are available in the ZenML Dashboard
- HTML visualizations can be automatically saved to configurable directories

## 📁 Project Organization

```
omni-reader/
│
├── configs/               # YAML configuration files
├── pipelines/             # ZenML pipeline definitions
├── steps/                 # Pipeline step implementations
├── utils/                 # Utility functions
│   ├── config.py          # Configuration utilities
│   └── ...
├── run.py                 # Main script for running the pipeline
├── config_generator.py    # Tool for generating experiment configurations
└── README.md              # Project documentation
```

## 🔗 Links

- [ZenML Documentation](https://docs.zenml.io/)
- [Mistral AI Vision Documentation](https://docs.mistral.ai/capabilities/vision/)
- [Gemma3 Documentation](https://ai.google.dev/gemma/docs/core)
