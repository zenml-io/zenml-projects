
---
title: ZenML Breast Cancer Classifier
emoji: 🦀
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: apache-2.0
---

# ZenML MLOps Breast Cancer Classification Demo

## 🌍 Project Overview

This is a minimalistic MLOps project demonstrating how to put machine learning 
workflows into production using ZenML. The project focuses on building a breast 
cancer classification model with end-to-end ML pipeline management.

### Key Features

- 🔬 Feature engineering pipeline
- 🤖 Model training pipeline
- 🧪 Batch inference pipeline
- 📊 Artifact and model lineage tracking
- 🔗 Integration with Weights & Biases for experiment tracking

## 🚀 Installation

1. Clone the repository
2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3. Install ZenML integrations:
    ```bash
    zenml integration install sklearn xgboost wandb -y
    zenml login
    zenml init
    ```
4. You need to register a stack with a [Weights & Biases Experiment Tracker](https://docs.zenml.io/stack-components/experiment-trackers/wandb). 

## 🧠 Project Structure

- `steps/`: Contains individual pipeline steps
- `pipelines/`: Pipeline definitions
- `run.py`: Main script to execute pipelines

## 🔍 Workflow and Execution

First, you need to set your stack:

```bash
zenml stack set stack-with-wandb
```

### 1. Data Loading and Feature Engineering

- Uses the Breast Cancer dataset from scikit-learn
- Splits data into training and inference sets
- Preprocesses data for model training

```bash
python run.py --feature-pipeline
```

### 2. Model Training

- Supports multiple model types (SGD, XGBoost)
- Evaluates and compares model performance
- Tracks model metrics with Weights & Biases

```bash
python run.py --training-pipeline
```

### 3. Batch Inference

- Loads production model
- Generates predictions on new data

```bash
python run.py --inference-pipeline
```
