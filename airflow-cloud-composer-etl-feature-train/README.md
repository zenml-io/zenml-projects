# ZenML ECB Interest Rate Project

This is a comprehensive machine learning project built with the ZenML framework and its integrations. The project trains XGBoost regression models to predict ECB interest rates based on historical data. This project was developed using ZenML's best practices for production-ready machine learning pipelines.

## 👋 Introduction

Welcome to the ECB Interest Rate Prediction project! This project demonstrates how to use ZenML to create a production-ready machine learning pipeline for predicting European Central Bank (ECB) interest rates. The project contains a collection of standard and custom ZenML steps, pipelines, and other artifacts that serve as a solid starting point for your journey with ZenML.

To get started, you can run the project as-is without any code changes. Here's how:

```bash
# Set up a Python virtual environment, if you haven't already
python3 -m venv .venv
source .venv/bin/activate
# Install requirements & integrations
pip install -r requirements.txt
# Start the ZenML UI locally (optional)
zenml up
# Run the pipeline included in the project
python run.py
```

After running the pipelines, you can check the results in the ZenML UI by following the link printed in the terminal.
Next steps:

- Explore the CLI options: python run.py --help
- Review the project structure and code
- Read the ZenML documentation to learn more about ZenML concepts
- Start customizing the project for your specific needs

## 📦 What's in the box?
This project demonstrates how to implement key steps of the ML Production Lifecycle using ZenML. It consists of three main pipelines:

- ETL Pipeline
- Feature Engineering Pipeline
- Model Training Pipeline

These pipelines work together to:

- Load and preprocess ECB interest rate data
- Engineer relevant features for interest rate prediction
- Train and evaluate XGBoost models
- Compare newly trained models with previously deployed models

The project uses the Model Control Plane to manage model versions and ensure that only quality-assured models are used for predictions.

## 📜 Project Structure

The project follows the recommended ZenML project structure:

├── configs                   # pipelines configuration files
│   ├── etl_develop.yaml      # configuration for ETL pipeline (development)
│   ├── etl_production.yaml   # configuration for ETL pipeline (production)
│   ├── feature_engineering_develop.yaml    # configuration for feature engineering (development)
│   ├── feature_engineering_production.yaml # configuration for feature engineering (production)
│   ├── training_develop.yaml # configuration for training pipeline (development)
│   └── training_production.yaml # configuration for training pipeline (production)
├── pipelines                 # `zenml.pipeline` implementations
│   ├── etl.py                # ETL pipeline
│   ├── feature_engineering.py # Feature Engineering pipeline
│   └── training.py           # Training Pipeline
├── steps                     # `zenml.steps` implementations
│   ├── data_loading.py       # steps for loading data
│   ├── feature_engineering.py # steps for feature engineering
│   ├── model_training.py     # steps for model training and evaluation
│   └── model_deployment.py   # steps for model deployment
├── utils                     # helper functions
├── README.md                 # this file
├── requirements.txt          # Python dependencies 
└── run.py                    # CLI tool to run pipelines

Feel free to modify and expand upon this project to suit your specific ECB interest rate prediction needs!