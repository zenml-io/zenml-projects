# Bank Subscription Prediction

A ZenML-based project for predicting bank term deposit subscriptions.

## Project Structure

```
bank_subscription_prediction/
├── configs/             # YAML Configuration files
│   ├── __init__.py
│   ├── baseline.yaml    # Baseline experiment config
│   ├── more_trees.yaml  # Config with more trees
│   └── deeper_trees.yaml# Config with deeper trees
├── pipelines/           # ZenML pipeline definitions
│   ├── __init__.py
│   └── training_pipeline.py
├── steps/               # ZenML pipeline steps
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── data_preprocessor.py
│   ├── data_splitter.py
│   ├── model_trainer.py
│   └── model_evaluator.py
├── utils/               # Utility functions and helpers
│   ├── __init__.py
│   └── model_utils.py
├── __init__.py
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── run.py               # Main script to run the pipeline
```

## Credits

This project is based on the Jupyter notebook [predict_bank_cd_subs_by_xgboost_clf_for_imbalance_dataset.ipynb](https://github.com/IBM/xgboost-financial-predictions/blob/master/notebooks/predict_bank_cd_subs_by_xgboost_clf_for_imbalance_dataset.ipynb) from IBM's xgboost-financial-predictions repository. The original work demonstrates XGBoost classification for imbalanced datasets and has been adapted into a complete ZenML pipeline.

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure ZenML is initialized:
   ```
   zenml init
   ```

## Dataset

This project uses the [Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) from the UCI Machine Learning Repository. The data loader will automatically download and cache the dataset if it's not available locally. No need to manually download the data!

## Running the Pipeline

### Basic Usage

```
python run.py
```

### Using Different Configurations

```
python run.py --config configs/more_trees.yaml
```

### Available Configurations

- `baseline.yaml`: Default XGBoost parameters
- `more_trees.yaml`: Increased number of estimators (200)
- `deeper_trees.yaml`: Increased maximum tree depth (5)

### Creating Custom Configurations

You can create new YAML configuration files by copying and modifying existing ones:

```yaml
# my_custom_config.yaml
# Start with copying an existing config and modify the values
# environment configuration
settings:
  docker:
    required_integrations:
      - sklearn
      - pandas
    requirements:
      - matplotlib
      - xgboost
      - seaborn
      - plotly
      - jupyter

# Model Control Plane config
model:
  name: bank_subscription_classifier
  version: 0.1.0
  license: MIT
  description: A bank term deposit subscription classifier
  tags: ["bank_marketing", "classifier", "xgboost"]

# Custom step parameters
steps:
  # ...other step params...
  train_xgb_model_with_feature_selection:
    n_estimators: 300
    max_depth: 4
    # ...other parameters...
```

## Pipeline Steps

1. **Data Loading**: Auto-download or load the bank marketing dataset
2. **Data Cleaning**: Handle missing values
3. **Data Preprocessing**: Process categorical variables, drop unnecessary columns
4. **Data Splitting**: Split data into training and test sets
5. **Model Training**: Train an XGBoost classifier with selected features
6. **Model Evaluation**: Evaluate model performance and visualize results with interactive HTML visualization

## Project Details

This project demonstrates how to:
- Handle imbalanced classification using XGBoost
- Implement feature selection 
- Create reproducible ML pipelines with ZenML
- Organize machine learning code in a maintainable structure
- Use YAML configurations for clean step parameterization
- Generate interactive HTML visualizations for model evaluation 