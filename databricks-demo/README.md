# ZenML Databricks Integration Demo

This project demonstrates how to use ZenML with Databricks for machine learning model training and deployment. It showcases the integration between ZenML's MLOps capabilities and Databricks' distributed computing platform.

## 👋 Introduction

This project serves as a practical example of building an end-to-end ML pipeline using ZenML and Databricks. It includes:

- Training ML models using Databricks' distributed computing capabilities
- MLflow integration for experiment tracking and model registry
- Model deployment and inference pipelines
- Integration with Databricks' managed MLflow service

## 🚀 Getting Started

To run this project:

```bash
# Set up a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements & integrations
make setup

# Configure your Databricks workspace
# (You'll need your Databricks workspace URL and access token)

# Run the pipeline
python run.py
```

## 🎛️ Command Line Options

The project provides several command-line flags to customize pipeline execution:

### Pipeline Selection
- `--training`: Run only the training pipeline
- `--deployment`: Run only the deployment pipeline
- `--inference`: Run only the batch inference pipeline

### Data Processing Options
- `--no-drop-na`: Skip dropping rows with missing values
- `--no-normalize`: Skip data normalization
- `--drop-columns COL1,COL2,...`: Comma-separated list of columns to drop
- `--test-size FLOAT`: Proportion of data for test set (default: 0.2)

### Model Quality Gates
- `--min-train-accuracy FLOAT`: Minimum required training accuracy (default: 0.8)
- `--min-test-accuracy FLOAT`: Minimum required test accuracy (default: 0.8)
- `--fail-on-accuracy-quality-gates`: Fail pipeline if accuracy thresholds aren't met

### Pipeline Execution
- `--no-cache`: Disable caching for the pipeline run

### Examples

```bash
# Run training pipeline with custom settings
python run.py --training --test-size 0.3 --min-train-accuracy 0.85

# Run inference pipeline without caching
python run.py --inference --no-cache

# Run training with strict quality gates
python run.py --training --min-train-accuracy 0.9 --min-test-accuracy 0.85 --fail-on-accuracy-quality-gates

# Run training with custom data preprocessing
python run.py --training --no-normalize --drop-columns feature1,feature2
```

## 📦 Project Features

The project consists of three main pipelines:

1. **Training Pipeline**: Trains a machine learning model using Databricks' distributed computing
   - Data preprocessing and feature engineering
   - Model training with MLflow tracking
   - Model evaluation and registration
   - Automatic model versioning and promotion
   - Quality gates for model performance

2. **Deployment Pipeline**: Deploys the trained model to Databricks
   - Model artifact deployment
   - Service configuration
   - Deployment validation

3. **Batch Inference Pipeline**: Runs batch predictions using the deployed model
   - Data preprocessing
   - Batch inference using Databricks compute
   - Results storage and logging
   - Data drift detection
   - Performance monitoring

## 📜 Project Structure

```
.
├── configs/                  # Pipeline configuration files
├── pipelines/               # ZenML pipeline implementations
├── steps/                   # Pipeline step implementations
│   ├── training/           # Model training steps
│   ├── deployment/         # Model deployment steps
│   └── inference/          # Batch inference steps
├── utils/                   # Helper utilities
├── requirements.txt         # Python dependencies
└── run.py                   # CLI tool to run pipelines
```

## 🔧 Configuration

The project uses YAML configuration files in the `configs/` directory:

### Training Configuration (`train_config.yaml`)
- Model hyperparameters
- Training environment settings
- MLflow experiment configuration
- Data preprocessing options

### Deployment Configuration (`deployer_config.yaml`)
- Databricks workspace settings
- Model serving configuration
- Endpoint configuration
- Resource allocation

### Inference Configuration (`inference_config.yaml`)
- Batch size settings
- Data drift thresholds
- Logging configuration
- Performance monitoring settings

## 🔌 Prerequisites

1. **Databricks Workspace**
   - Active Databricks workspace
   - Workspace URL
   - Access token with appropriate permissions

3. **Required Stack Components**
   - MLflow experiment tracker
   - Databricks orchestrator
   - Model registry (optional)

## 🤝 Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.

## 📝 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
