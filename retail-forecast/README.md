# RetailForecast – Retail Demand Forecasting with ZenML

RetailForecast is a production-ready ML pipeline for retail demand forecasting, built with the ZenML framework. This project demonstrates how to build an end-to-end time series forecasting solution for retail applications using state-of-the-art deep learning models.

## Features

- **Synthetic Retail Data Generation**: Built-in capability to generate realistic retail sales data with patterns like seasonality, promotions, and holidays
- **State-of-the-Art Forecasting**: Uses Temporal Fusion Transformer (TFT), a powerful and interpretable model for multi-horizon forecasting
- **Production-Ready Pipeline**: Complete ZenML pipeline from data loading to model evaluation and prediction
- **Comprehensive Evaluation**: Calculates key retail metrics (MAE, RMSE, MAPE, SMAPE) and provides visualizations
- **Extensible Architecture**: Easily swap components (data sources, models, etc.) for your specific needs
- **Dockerized Deployment**: Ready to run locally or deploy to any cloud infrastructure

## Quick Start

```bash
# 1. Create & activate a clean Python >=3.9 environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize ZenML
zenml init

# 4. Run the full pipeline (training + inference)
python run.py

# 5. Run only training or prediction
python run.py --train
python run.py --predict

# 6. Run with custom parameters 
python run.py --train --forecast-horizon 28 --epochs 20 --batch-size 128 --hidden-size 128

# 7. Disable caching
python run.py --train --no-cache
```

## Project Structure

```
retailforecast/
├── pipelines/                  # ZenML pipeline definitions
│   ├── training_pipeline.py    # Training pipeline
│   └── inference_pipeline.py   # Inference pipeline
├── steps/                      # Individual ZenML step implementations
│   ├── data_loader.py          # Loads or generates synthetic data
│   ├── data_validator.py       # Validates data quality
│   ├── data_preprocessor.py    # Feature engineering for time series
│   ├── model_trainer.py        # Trains TFT model
│   ├── model_evaluator.py      # Evaluates model performance
│   └── predictor.py            # Generates future forecasts
├── data/                       # Data directory (synthetic data will be saved here)
├── Dockerfile                  # For containerized execution
├── requirements.txt            # Python dependencies
└── run.py                      # Script to run pipelines with options
```

## The Forecasting Pipeline

This project implements a comprehensive forecasting pipeline:

1. **Data Loading/Generation**: Either loads retail data from CSV files or generates synthetic data with realistic retail patterns (weekends, holidays, promotions, etc.)

2. **Data Validation**: Checks for common data quality issues in retail datasets (missing values, negative sales, duplicates) and fixes them

3. **Data Preprocessing**: 
   - Creates time-based features (day of week, month, etc.)
   - Generates lag features and rolling statistics
   - Splits data into train/validation/test sets by time
   - Normalizes features for neural networks

4. **Model Training**: Trains a Temporal Fusion Transformer (TFT) model
   - Multi-horizon, multi-series forecasting
   - Attention mechanism for handling long-term dependencies
   - Interpretable variable importance

5. **Model Evaluation**:
   - Computes key retail metrics (MAE, RMSE, MAPE, SMAPE)
   - Generates visualizations of forecasts
   - Identifies problematic store/item combinations

6. **Future Forecasting**:
   - Generates forecasts for a specified horizon
   - Creates visualizations of predictions
   - Exports forecasts to CSV for business use

## Customization Options

### Using Your Own Data

Place your sales data in CSV format in the `data/` directory:
- `sales.csv` with columns: date, store, item, sales
- `calendar.csv` with columns: date, is_weekend, is_holiday, is_promo, etc.

### Extending the Model

To use another forecasting model:
1. Modify `steps/model_trainer.py` to implement a different model (e.g., DeepAR)
2. Update the evaluation and prediction steps accordingly

### Deployment Options

For production deployment:

```bash
# Build the Docker image
docker build -t retailforecast:latest .

# Run with Docker
docker run -it retailforecast:latest

# Deploy to a ZenML stack
zenml stack register custom_stack -a <artifacts> -o <orchestrator> -c <container-registry>
zenml stack set custom_stack
python run.py --train
```

## Advanced Usage

### Hyperparameter Optimization

To find optimal model parameters:

```bash
# Conduct a search over hyperparameters
python run.py --train --epochs 5  # Use fewer epochs for quick iteration
```

Modify the `train_model` step to implement hyperparameter optimization with Optuna.

### Drift Detection

For a production environment, consider:
1. Implementing a separate drift detection pipeline
2. Setting automated retraining triggers
3. Adding model versioning

## Technical Details

The implemented TFT model:
- Combines LSTM encoders with self-attention mechanisms
- Learns hierarchical patterns across store-item combinations
- Provides interpretable attention scores for explainability
- Outputs probabilistic forecasts with prediction intervals

## License

This project is licensed under Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- The TFT model is based on the paper ["Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"](https://arxiv.org/abs/1912.09363) by Lim et al.
- Implements features inspired by the M5 Forecasting Competition best practices
