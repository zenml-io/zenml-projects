# Zen OCR

Zen OCR is a document processing workflow that ingests unstructured documents (PDFs, images, scans), extracts text via OCR, and uses Large Language Models (LLMs) to derive structured insights. ZenML's pipeline framework ensures the solution is **reproducible, automated, and cloud-agnostic**, leveraging its strengths like integration flexibility, artifact tracking, and environment reproducibility.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and configured for Gemma3
- Mistral API key (set as environment variable `MISTRAL_API_KEY`)
- ZenML installed and configured

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ocr-king.git
cd ocr-king

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

The app provides two execution modes:
- **Fast mode** (default): Uses direct function calls without ZenML overhead for quick interactive use
- **ZenML tracking mode**: Uses ZenML steps with full tracking capabilities (slower but provides pipeline visualizations)

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
- Run `zenml up` to access the dashboard
- MLflow integration provides detailed metrics and artifacts

## 🔗 Links

- [ZenML Documentation](https://zenml.io/docs)
- [Mistral AI](https://mistral.ai/)
- [Gemma3 Documentation](https://ai.google.dev/gemma3)
