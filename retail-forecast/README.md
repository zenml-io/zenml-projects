# Simple Retail Forecasting with ZenML and Prophet

This project demonstrates a minimalist but effective retail sales forecasting pipeline using ZenML and Facebook Prophet. It showcases the value of ML pipelines while keeping implementation complexity low.

## Features

- **Simple Time Series Forecasting**: Uses Facebook Prophet, a robust and easy-to-use forecasting tool
- **Tracked ML Pipeline**: All steps are tracked and logged with ZenML
- **Reproducible Experiments**: Each run is versioned and can be compared
- **Visual Reports**: Automatically generates forecast plots and metrics

## Pipeline Steps

1. **Data Loading**: Loads (or generates) synthetic retail sales data
2. **Data Preprocessing**: Prepares data for Prophet and creates train/test splits
3. **Model Training**: Trains a Prophet model for each store-item combination
4. **Model Evaluation**: Evaluates models on test data and logs metrics
5. **Forecasting**: Generates future forecasts and visualizations

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Initialize ZenML (if you haven't already):

```bash
zenml init
```

### Running the Pipeline

To run the pipeline with default parameters:

```bash
python run.py
```

With custom parameters:

```bash
python run.py --forecast-periods 60 --test-size 0.3 --weekly-seasonality True
```

### Viewing Results

Start the ZenML dashboard:

```bash
zenml up
```

Then follow the link to the dashboard where you can view:
- Pipeline runs
- Metrics and visualizations
- Artifacts (trained models and datasets)

## Pipeline Value Proposition

This simplified pipeline demonstrates several key benefits:

1. **Reproducibility**: Every experiment is tracked, including code, data, and parameters
2. **Collaboration**: Team members can share and reuse pipeline components
3. **Governance**: All models and their performance metrics are tracked
4. **Deployment**: Pipelines can be easily connected to deployment infrastructure
5. **Observability**: Track data and model performance over time

## Customization

To use your own data:
1. Place a CSV file at `data/sales.csv` with columns: date, store, item, sales
2. Run the pipeline - it will automatically use your data instead of synthetic data

## License

This project is licensed under the Apache License 2.0.
