# Databricks + ZenML: End-to-End Explainable ML Project

Welcome to this end-to-end demo project that showcases how to train, deploy, and run batch inference on a machine learning model using ZenML in a Databricks environment. This setup demonstrates how ZenML can simplify the end-to-end process of building reproducible, production-grade ML pipelines with minimal fuss.

## Overview

This project uses an example classification dataset (Breast Cancer) and provides three major pipelines:

1. Training Pipeline
2. Deployment Pipeline
3. Batch Inference Pipeline (with SHAP-based model explainability)

The pipelines are orchestrated via ZenML. Additionally, this setup uses:
- Databricks as the orchestrator
- MLflow for experiment tracking and model registry
- Evidently for data drift detection
- SHAP for model explainability during inference
- Slack notifications (configurable through ZenML's alerter stack components)

## Why ZenML?

ZenML is a lightweight MLOps framework for reproducible pipelines. With ZenML, you get:

- A consistent, standardized way to develop, version, and share pipelines.  
- Easy integration with various cloud providers, experiment trackers, model registries, and more.  
- Reproducibility and better collaboration: your pipelines and associated artifacts are automatically tracked and versioned.  
- Simple command-line interface for spinning pipelines up and down with different stack components (like local or Databricks orchestrators).  
- Built-in best practices for production ML, including quality gates for data drift and model performance thresholds.

## Project Structure

Here's an outline of the repository:

```
.
├── configs                   # Pipeline configuration files
│   ├── deployer_config.yaml  # Deployment pipeline config
│   ├── inference_config.yaml # Batch inference pipeline config
│   └── train_config.yaml     # Training pipeline config
├── pipelines                 # ZenML pipeline definitions
│   ├── batch_inference.py    # Orchestrates batch inference
│   ├── deployment.py         # Deploys a model service
│   └── training.py           # Trains and promotes model
├── steps                     # ZenML steps logic
│   ├── alerts                # Alert/notification logic
│   ├── data_quality          # Data drift and quality checks
│   ├── deployment            # Deployment step
│   ├── etl                   # ETL steps (data loading, preprocessing, splitting)
│   ├── explainability        # SHAP-based model explanations
│   ├── hp_tuning             # Hyperparameter tuning pipeline steps
│   ├── inference             # Batch inference step
│   ├── promotion             # Model promotion logic
│   └── training              # Model training and evaluation steps
├── utils                     # Helper modules
├── Makefile                  # Quick integration setup commands
├── requirements.txt          # Python dependencies
├── run.py                    # CLI to run pipelines
└── README.md                 # This file
```

## Getting Started

1. (Optional) Create and activate a Python virtual environment:  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:  
   ```bash
   make setup
   ```
   This installs the required ZenML integrations (MLflow, Slack, Evidently, Kubeflow, Kubernetes, AWS, etc.) and any library dependencies.

3. (Optional) Set up a local Stack (if you want to try this outside Databricks):  
   ```bash
   make install-stack-local
   ```

4. If you have Databricks properly configured in your ZenML stack (with the Databricks token secret set up, cluster name, etc.), you can orchestrate the pipelines on Databricks by default.

## Running the Project

All pipeline runs happen via the CLI in run.py. Here are the main options:

• View available options:  
  ```bash
  python run.py --help
  ```

• Run everything (train, deploy, inference) with default settings:  
  ```bash
  python run.py --training --deployment --inference
  ```
  This will:
  1. Train a model and evaluate its performance
  2. Deploy the model if it meets quality criteria
  3. Run batch inference with SHAP explanations and data drift checks

• Run just the training pipeline (to build or update a model):  
  ```bash
  python run.py --training
  ```

• Run just the deployment pipeline (to deploy the latest staged model):  
  ```bash
  python run.py --deployment
  ```

• Run just the batch inference pipeline (to generate predictions and explanations while checking for data drift):  
  ```bash
  python run.py --inference
  ```

### Additional Command-Line Flags

• Disable caching:  
  ```bash
  python run.py --no-cache --training
  ```

• Skip dropping NA values or skipping normalization:  
  ```bash
  python run.py --no-drop-na --no-normalize --training
  ```

• Drop specific columns:  
  ```bash
  python run.py --training --drop-columns columnA,columnB
  ```

• Set minimal accuracy thresholds for training and test sets:  
  ```bash
  python run.py --min-train-accuracy 0.9 --min-test-accuracy 0.8 --fail-on-accuracy-quality-gates --training
  ```

When you run any of these commands, ZenML will orchestrate each pipeline on the active stack (Databricks if configured) and log the results in your model registry (MLflow). If you have Slack or other alerter components configured, you'll see success/failure notifications.

## Observing Your Pipelines

ZenML offers a local dashboard that you can launch with:
```bash
zenml up
```
Check the terminal logs for the local web address (usually http://127.0.0.1:8237). You'll see pipeline runs, steps, and artifacts.  

If you deployed on Databricks, you can also see the runs orchestrated in the Databricks jobs UI. The project is flexible enough to run the same pipelines locally or in the cloud without changing the code.

## Contributing & License

Contributions and suggestions are welcome. This project is licensed under the Apache License 2.0.  

For questions, feedback, or support, please reach out to the ZenML community or open an issue in this repository.

---
