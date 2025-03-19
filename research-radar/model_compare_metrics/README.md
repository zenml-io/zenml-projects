# Model Comparison Metrics

This directory contains JSON files that store performance metrics and configuration data from the model comparison runs between ModernBERT and Claude Haiku.

## File Structure

Files follow the naming convention: `comparison_metrics_YYYYMMDD_HHMMSS.json`

Each JSON file contains:

- A `config` object with the following fields:

  - `sample_size`: Sample size for the comparison run
  - `modernbert_batch_size`: Batch size for ModernBERT
  - `claude_batch_size`: Batch size for Claude Haiku
  - `random_seed`: Random seed used

- A `metrics` object with the following fields:

  - `modernbert`: Metrics for ModernBERT
  - `claude`: Metrics for Claude Haiku

  Each model object contains:

  - `performance`: Performance metrics (accuracy, F1, precision, recall)
  - `avg_latency`: Average latency
  - `cost_per_1000`: Cost per 1000 samples
  - `samples_processed`: Number of samples processed

Claude-specific metrics:

- `avg_tokens`: Average tokens per request
- `error_rate`: Error rate

Example:

```json
{
  "config": {
    "sample_size": 50,
    "modernbert_batch_size": 15,
    "claude_batch_size": 5,
    "random_seed": 42
  },
  "metrics": {
    "modernbert": {
      "performance": {
        "accuracy": 1.0,
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0
      },
      "avg_latency": 0.096,
      "cost_per_1000": 1.913,
      "samples_processed": 50
    },
    "claude": {
      "performance": {
        "accuracy": 0.76,
        "f1": 0.76,
        "precision": 0.76,
        "recall": 0.76
      },
      "avg_latency": 4.227,
      "avg_tokens": 2839.64,
      "error_rate": 0.0,
      "cost_per_1000": 283.964,
      "samples_processed": 50
    }
  },
  "timestamp": "20250219_011816"
}
```
