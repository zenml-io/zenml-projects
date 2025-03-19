# Pipeline Usage

- For detailed implementation of each step, see the individual Python files
- For pipeline configurations and settings, refer to [`base_config.yaml`](../base_config.yaml)

## Classification Pipeline

Runs the following steps:

- [`load_classification_dataset`](../steps/load_classification_dataset.py) - Loads articles based on classification mode
- [`classify_articles`](../steps/classify_articles.py) - Classifies articles using DeepSeek R1
- [`save_classifications`](../steps/save_classifications.py) - Saves classification results to JSON
- [`merge_classifications`](../steps/merge_classifications.py) - Merges new classifications with existing dataset (augmentation mode)

**Usage modes**:
- `evaluation`: Tests classification on existing labeled articles
- `augmentation`: Processes new articles to expand the training dataset

## Training Pipeline

Runs the following steps:

- [`load_training_dataset`](../steps/load_training_dataset.py) - Automatically selects augmented dataset if available, otherwise uses composite dataset
- [`data_preprocessor`](../steps/data_preprocessor.py) - Prepares text for model training
- [`data_splitter`](../steps/data_splitter.py) - Creates train/validation/test splits
- [`save_test_set`](../steps/save_test_set.py) - Optionally saves test set for later evaluation
- [`finetune_modernbert`](../steps/finetune_modernbert.py) - Fine-tunes the ModernBERT model
- [`save_model_local`](../steps/save_model_local.py) - Saves model and tokenizer artifacts

## Deployment Pipeline

Runs the following step:

- [`push_model_to_huggingface`](../steps/push_model_to_huggingface.py)

## Comparison Pipeline

Runs the following steps:

- [`load_test_set_from_artifact`](../steps/load_test_set_from_artifact.py)
- [`compare_models`](../steps/compare_models.py)
- [`save_comparison_metrics`](../steps/save_comparison_metrics.py)
