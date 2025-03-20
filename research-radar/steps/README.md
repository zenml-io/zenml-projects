# Steps Directory

This directory contains the individual pipeline steps used across the project.

## Classification Pipeline Steps

### Data Loading

- [`data_loader.py`](data_loader.py): Contains three loading functions:
  - `load_classification_dataset()`: Loads articles for DeepSeek classification
  - `load_training_dataset()`: Loads combined dataset for training

### Classification

- [`classify_articles.py`](classify_articles.py): Performs article classification using DeepSeek R1
  - Supports both batch processing and full dataset processing
  - Implements parallel processing with configurable workers
  - Includes progress tracking for long-running classification tasks
  - Uses retry logic for API reliability
- [`save_classifications.py`](save_classifications.py): Saves classification results in structured JSON format
  - Adapts file naming based on batch processing parameters
- [`merge_classifications.py`](merge_classifications.py): Merges new DeepSeek classifications with existing dataset

### Classification Helpers

- [`utils/classification_helpers.py`](../utils/classification_helpers.py): Contains utilities for article classification
  - Core logic for interacting with the HuggingFace inference API
  - Error handling and retry mechanisms
  - Shared functionality for both sequential and parallel processing

## Training Pipeline Steps

### Data Processing

- [`data_preprocessor.py`](data_preprocessor.py): Standardizes text inputs and label encoding
- [`data_splitter.py`](data_splitter.py): Performs stratified dataset splitting (train/validation/test)

### Model Training

- [`finetune_modernbert.py`](finetune_modernbert.py): Fine-tunes ModernBERT with:
  - Cosine LR schedule with warmup
  - Early stopping on F1 score
  - Performance metrics logging
  - Model checkpointing

### Artifact Management

- [`save_model_local.py`](save_model_local.py): Exports model and tokenizer artifacts locally to the `models` directory
- [`save_test_set.py`](save_test_set.py): Saves test dataset for later evaluation to the `artifacts` directory
- [`load_test_set.py`](load_test_set.py): Loads test dataset from saved artifact (uses `load_artifact` from ZenML)

## Deployment Pipeline Steps

- [`push_model_to_huggingface.py`](push_model_to_huggingface.py): Deploys model and tokenizer to HuggingFace Hub (currently set to my HuggingFace account/repo)

## Comparison Pipeline Steps

- [`compare_models.py`](compare_models.py): Evaluates ModernBERT against Claude Haiku:
  - Performance metrics (accuracy, F1, etc.)
  - Latency measurements
  - Cost analysis
- [`save_comparison_metrics.py`](save_comparison_metrics.py): Saves comparison results as structured JSON to the `model_compare_metrics` directory
