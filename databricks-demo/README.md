# ZenML Production Line QA Project

This repository provides an end-to-end (E2E) example of using ZenML to build, deploy, and monitor a production-quality machine learning pipeline for quality assurance (QA). By leveraging ZenML, you can define clear pipeline steps, incorporate best practices around modularity and reusability, and integrate with powerful external tools such as MLflow for experiment tracking, Evidently for data/quality checks, and Databricks for scalable compute.

## Introduction

This project demonstrates how to:
- Retrieve and split data into train and inference sets
- Execute data preprocessing and engineering
- Perform hyperparameter tuning to find the best model
- Automatically train, evaluate, and deploy the best-performing model
- Run batch inference, complete with data drift checks and optional notifications

We use the Breast Cancer dataset to illustrate the flow of data and the concept of model performance checks for QA.

## Project Features

1. **Training Pipeline**  
   - Data loading, splitting, and preprocessing  
   - Hyperparameter tuning  
   - Best model selection and evaluation  
   - Optional quality gates for model performance  
   - Automatic promotion logic based on prior model metrics

2. **Deployment Pipeline**  
   - Deploys the validated model for inference  
   - Includes integration with Databricks for scalable serving  
   - Configuration for resource sizing

3. **Batch Inference Pipeline**  
   - Runs batch predictions using the deployed model  
   - Integration with Evidently to detect data drift  
   - Notifies on success/failure based on configuration

## Getting Started

1. Clone this repository and navigate into the project directory:
   ```bash
   git clone <your_fork_or_clone_url_here>
   cd databricks-demo
   ```

2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   
3. Install requirements and any relevant ZenML integrations:
   ```bash
   make setup
   ```
   This command runs "pip install -r requirements.txt" and installs needed ZenML integrations (AWS, MLflow, Slack, Databricks, etc.).

4. Configure your Databricks workspace and ensure you have a valid stack set in ZenML. For example:
   ```bash
   # Optionally configure your stack, e.g.:
   zenml integration install databricks mlflow evidently -y
   # or copy/paste the lines from "Makefile" that install integrations
   
   # Set up the local or remote stack
   make install-stack-local
   ```

## How to Run

Use the provided CLI tool (run.py) in combination with the command-line flags to customize how the pipelines execute.

Examples:
```bash
# Run the training pipeline (with default parameters)
python run.py --training

# Run only batch inference pipeline
python run.py --inference

# Run the deployment pipeline
python run.py --deployment

# Disable ZenML caching for any run
python run.py --no-cache --training

# Customize data preprocessing
python run.py --training --no-drop-na --no-normalize --drop-columns colA,colB

# Enforce minimum quality gates for training/test accuracy
python run.py --training --min-train-accuracy 0.9 --min-test-accuracy 0.85 --fail-on-accuracy-quality-gates
```

### Command-Line Flags
- `--training`: Run only the training pipeline.  
- `--deployment`: Run only the deployment pipeline.  
- `--inference`: Run only the batch inference pipeline.  
- `--no-cache`: Disable pipeline step caching.  
- `--no-drop-na`: Skip dropping rows containing NA values.  
- `--no-normalize`: Skip MinMaxScaler-based normalization.  
- `--drop-columns COL1,COL2,...`: Drop the specified columns before training.  
- `--test-size FLOAT`: Proportion of the dataset used for testing (default: 0.2).  
- `--min-train-accuracy FLOAT`: Minimum training accuracy threshold.  
- `--min-test-accuracy FLOAT`: Minimum test accuracy threshold.  
- `--fail-on-accuracy-quality-gates`: If accuracy thresholds aren't met, fail early.  

## Why ZenML?

ZenML provides an opinionated yet flexible approach to building production ML pipelines. It integrates seamlessly with:
- Databricks for scalable computing
- MLflow for experiment tracking and model registry  
- Slack or other alert channels for success/failure notifications  
- Data validation frameworks (like Evidently) for drift, data quality, or fairness checks  

With ZenML, you keep your pipeline logic clean and easily versionable while plugging in your favorite tooling.

## Contributing

Feel free to open issues or create pull requests for improvements or bug fixes. We welcome community submissions!

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
