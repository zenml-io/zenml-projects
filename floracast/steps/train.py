"""
Model training step for FloraCast.
"""

import os
import pickle
from typing import Dict, Any, Tuple, Annotated
from darts import TimeSeries
from darts.models import TFTModel, ExponentialSmoothing
from zenml import step
from zenml.logger import get_logger
from materializers import TFTModelMaterializer  # Import to register the materializer

logger = get_logger(__name__)


@step
def train_model(
    train_series: Annotated[TimeSeries, "train_series"],
    model_name: str = "TFTModel",
    input_chunk_length: int = 30,
    output_chunk_length: int = 7,
    hidden_size: int = 32,
    lstm_layers: int = 1,
    num_attention_heads: int = 2,
    dropout: float = 0.1,
    batch_size: int = 16,
    n_epochs: int = 5,
    random_state: int = 42,
    add_relative_index: bool = True,
    enable_progress_bar: bool = False,
    enable_model_summary: bool = False
) -> Tuple[
    Annotated[object, "trained_model"],
    Annotated[str, "artifact_uri"],
    Annotated[str, "model_class"],
]:
    """
    Train a forecasting model.
    
    Args:
        train_series: Training time series
        model_name: Name of the model class to use
        input_chunk_length: Number of time steps to use as input
        output_chunk_length: Number of time steps to predict
        hidden_size: Size of hidden layers
        lstm_layers: Number of LSTM layers
        num_attention_heads: Number of attention heads
        dropout: Dropout rate
        batch_size: Training batch size
        n_epochs: Number of training epochs
        random_state: Random seed
        add_relative_index: Whether to add relative index
        enable_progress_bar: Whether to show progress bar
        enable_model_summary: Whether to show model summary
        
    Returns:
        Tuple of (fitted_model, artifact_uri, model_class)
    """
    # Build model parameters dict
    model_params = {
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'hidden_size': hidden_size,
        'lstm_layers': lstm_layers,
        'num_attention_heads': num_attention_heads,
        'dropout': dropout,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'random_state': random_state,
        'add_relative_index': add_relative_index,
        'pl_trainer_kwargs': {
            'enable_progress_bar': enable_progress_bar,
            'enable_model_summary': enable_model_summary
        }
    }
    
    logger.info(f"Training {model_name} with params: {model_params}")
    
    # Initialize model based on configuration
    if model_name == 'TFTModel':
        model = TFTModel(**model_params)
    elif model_name == 'ExponentialSmoothing':
        # Fallback model with simpler parameters
        fallback_params = {k: v for k, v in model_params.items() 
                          if k in ['seasonal_periods', 'trend', 'seasonal']}
        model = ExponentialSmoothing(**fallback_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    logger.info(f"Starting training with {len(train_series)} data points")
    
    # Train the model
    try:
        model.fit(train_series)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Fallback to simpler model
        logger.info("Falling back to ExponentialSmoothing model")
        model = ExponentialSmoothing()
        model.fit(train_series)
        model_name = 'ExponentialSmoothing'
    
    # Save model artifacts
    os.makedirs('model_artifacts', exist_ok=True)
    model_path = f'model_artifacts/model_{model_name.lower()}.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Get absolute path for artifact URI
    artifact_uri = os.path.abspath(model_path)
    
    logger.info(f"Model saved to: {artifact_uri}")
    
    return model, artifact_uri, model_name