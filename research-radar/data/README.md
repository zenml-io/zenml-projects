# Data Directory

This directory contains the core datasets used in the article classification pipeline, stored in JSONL format.

## Files

### `composite_dataset.jsonl`

Contains a 70 human-verified LLMOps articles, manually classified as positive (`answer: accept`) for the LLMOps Database by Alex Strick van Linschoten (github: `strickvl`), as well as 70 negative (`answer: reject`) articles, for a total of 140 articles. You may use this dataset for the training pipeline directly if you wish to skip the classification pipeline.

This dataset is used for finetuning ModernBERT and contains the following fields:

- `text`: Article content
- `answer`: Classification result ("accept" or "reject")
- `meta["title"]`: Article title
- `meta["url"]`: Source URL
- `meta["published_date"]`: Publication date
- `meta["author"]`: Author information

### `unclassified_dataset.jsonl`

Contains 100 unclassified articles that can be used to augment the training data with new articles in the classification pipeline.

## Dataset Usage

The datasets are used at different stages of the pipeline:

1. `composite_dataset.jsonl` → Evaluation & Prompt Refinement (Classification Pipeline)
2. `unclassified_dataset.jsonl` → Augmentation (Classification Pipeline)
3. `augmented_dataset.jsonl` → Model Training (Training Pipeline)

For more details on how these datasets are processed, see:

- `steps/data_loader.py` for dataset loading logic
- `classification_results/README.md` for classification output details
