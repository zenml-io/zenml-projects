# Classification Results

This directory contains classification results from DeepSeek R1 model runs, organized into two main subdirectories for different stages of the classification pipeline.

## Directory Structure

```
classification_results/
├── for_augmentation/
│   └── YYYYMMDD_HHMMSS/
│       └── results.json
└── for_evaluation/
    └── YYYYMMDD_HHMMSS/
        ├── results.json
        ├── metrics.json
        └── report.md
```

## Subdirectories

#### `for_augmentation/`

Contains classification results from DeepSeek R1 from unclassified datasets (e.g. `unclassified_dataset.jsonl`) when running the classification pipeline in `augmentation` mode. These results can then be used to augment the training dataset with new articles.

### `for_evaluation/`

Contains classification results from DeepSeek R1 on the `composite_dataset.jsonl` or other datasets used in evaluation mode. Each run creates a timestamped directory containing the results, metrics, and report files.

## File Types

### `results.json`

Contains the complete classification results with:

- Timestamp and metadata (batch information)
- Model ID and complete inference parameters
- Classification results with associated metadata
- Metrics data (for evaluation runs, embedded in the same file)

### `metrics.json`

A separate copy of the metrics data including:

- Error analysis (count and types of errors)
- Performance metrics (accuracy, F1, precision, recall)
- Class distribution statistics
- Confidence analysis by prediction type

### `report.md`

A human-readable markdown report of the metrics data with formatted tables for:

- Valid/error sample counts and percentages
- Detailed error analysis with counts by error type
- Performance metrics (accuracy, F1, precision, recall, specificity, NPV)
- Class distribution statistics
- Confidence analysis by prediction type (TP, TN, FP, FN)

## Checkpoint System

The classification process uses a checkpoint system to enable resuming interrupted jobs:

- Checkpoints are saved periodically during processing in the `checkpoints/` directory
- The frequency and retention settings are configurable in `settings.yaml`
- Only the most recent N checkpoints are kept to conserve disk space

## Processing Configuration

Classification processing can be configured in [`settings.yaml`](../settings.yaml):

- Process entire dataset or specific batches
- Enable/disable parallel processing
- Configure number of worker threads
- Adjust inference parameters
- Control checkpoint behavior (frequency, resume capability, retention)

## Related Modules

- [`utils/classification_metrics.py`](../utils/classification_metrics.py): Calculates agreement metrics between ground truth and model classifications
- [`utils/checkpoint.py`](../utils/checkpoint.py): Manages the checkpoint system for resumable processing
- [`utils/classification_helpers.py`](../utils/classification_helpers.py): Contains core classification logic for the DeepSeek model
- [`steps/save_classifications.py`](../steps/save_classifications.py): Pipeline step that orchestrates classification saving
