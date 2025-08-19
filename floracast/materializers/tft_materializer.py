"""
Custom materializer for Darts TFT model objects using checkpoint-based serialization.
"""

import os
import json
import tempfile
import pickle
from typing import Type, Any, Dict

from darts.models import TFTModel
from zenml.enums import ArtifactType
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.io import fileio
from zenml.logger import get_logger

logger = get_logger(__name__)


class TFTModelMaterializer(BaseMaterializer):
    """Materializer for Darts TFT model objects using checkpoint-based serialization."""
    
    ASSOCIATED_TYPES = (TFTModel,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL
    
    def load(self, data_type: Type[Any]) -> TFTModel:
        """Load a TFT model using checkpoint-based approach."""
        
        # Method 1: Try to load from TFT's native save format
        model_dir_path = os.path.join(self.uri, "tft_model")
        
        if fileio.isdir(model_dir_path):
            logger.info("Attempting to load TFT model from native save format")
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    local_path = os.path.join(temp_dir, "tft_model")
                    fileio.copy(model_dir_path, local_path, recursive=True)
                    
                    # Load using TFT's native method
                    model = TFTModel.load(local_path)
                    logger.info("Successfully loaded TFT model from native format")
                    return model
            except Exception as e:
                logger.warning(f"Failed to load from native format: {e}")
        
        # Method 2: Try checkpoint-based loading
        checkpoint_path = os.path.join(self.uri, "model_checkpoint.ckpt")
        params_path = os.path.join(self.uri, "model_params.json")
        
        if fileio.exists(checkpoint_path) and fileio.exists(params_path):
            logger.info("Attempting to load TFT model from checkpoint")
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    local_checkpoint = os.path.join(temp_dir, "model.ckpt")
                    local_params = os.path.join(temp_dir, "params.json")
                    
                    # Download checkpoint and params
                    with fileio.open(checkpoint_path, "rb") as src:
                        with open(local_checkpoint, "wb") as dst:
                            dst.write(src.read())
                    
                    with fileio.open(params_path, "r") as f:
                        params = json.load(f)
                    
                    # Recreate model with same parameters
                    model = TFTModel(**params)
                    
                    # Load the checkpoint
                    model.load_from_checkpoint(local_checkpoint)
                    logger.info("Successfully loaded TFT model from checkpoint")
                    return model
                    
            except Exception as e:
                logger.warning(f"Failed to load from checkpoint: {e}")
        
        # Method 3: Fallback to pickle (with warning)
        pickle_path = os.path.join(self.uri, "model.pkl")
        
        if fileio.exists(pickle_path):
            logger.warning("Falling back to pickle deserialization (may have issues)")
            try:
                with fileio.open(pickle_path, "rb") as f:
                    model = pickle.load(f)
                
                # Try to restore model state if it's None
                if hasattr(model, 'model') and model.model is None:
                    logger.warning("Model.model is None, attempting to recreate...")
                    # This is expected to fail, but we try anyway
                    try:
                        model._create_model()
                    except:
                        logger.error("Could not recreate model - inference will fail")
                
                return model
            except Exception as e:
                logger.error(f"Pickle fallback failed: {e}")
        
        raise RuntimeError("Could not load TFT model using any available method")
    
    def save(self, data: TFTModel) -> None:
        """Save a TFT model using multiple serialization strategies."""
        
        logger.info("Saving TFT model using multiple strategies")
        
        # Method 1: Use TFT's native save method
        model_dir_path = os.path.join(self.uri, "tft_model")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = os.path.join(temp_dir, "tft_model")
                
                # Save using TFT's native method
                data.save(local_path)
                
                # Upload the entire model directory
                fileio.copy(local_path, model_dir_path, recursive=True)
                logger.info("Successfully saved TFT model in native format")
        except Exception as e:
            logger.warning(f"Native save failed: {e}")
        
        # Method 2: Save checkpoint and parameters separately
        if hasattr(data, 'model') and data.model is not None:
            try:
                checkpoint_path = os.path.join(self.uri, "model_checkpoint.ckpt")
                params_path = os.path.join(self.uri, "model_params.json")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    local_checkpoint = os.path.join(temp_dir, "model.ckpt")
                    
                    # Save PyTorch Lightning checkpoint
                    data.trainer.save_checkpoint(local_checkpoint)
                    
                    # Upload checkpoint
                    with open(local_checkpoint, "rb") as src:
                        with fileio.open(checkpoint_path, "wb") as dst:
                            dst.write(src.read())
                    
                    # Save model parameters
                    params = {
                        'input_chunk_length': data.input_chunk_length,
                        'output_chunk_length': data.output_chunk_length,
                        'hidden_size': getattr(data, 'hidden_size', 32),
                        'lstm_layers': getattr(data, 'lstm_layers', 1),
                        'num_attention_heads': getattr(data, 'num_attention_heads', 2),
                        'dropout': getattr(data, 'dropout', 0.1),
                        'batch_size': getattr(data, 'batch_size', 16),
                        'n_epochs': getattr(data, 'n_epochs', 5),
                        'random_state': getattr(data, 'random_state', 42),
                        'add_relative_index': getattr(data, 'add_relative_index', True),
                        'pl_trainer_kwargs': getattr(data, 'pl_trainer_kwargs', {}),
                    }
                    
                    with fileio.open(params_path, "w") as f:
                        json.dump(params, f, indent=2)
                    
                    logger.info("Successfully saved TFT model checkpoint and parameters")
            except Exception as e:
                logger.warning(f"Checkpoint save failed: {e}")
        
        # Method 3: Pickle fallback (always save for compatibility)
        try:
            pickle_path = os.path.join(self.uri, "model.pkl")
            with fileio.open(pickle_path, "wb") as f:
                pickle.dump(data, f)
            logger.info("Successfully saved TFT model as pickle backup")
        except Exception as e:
            logger.warning(f"Pickle backup failed: {e}")
            
        logger.info("TFT model saved using multiple strategies for maximum compatibility")