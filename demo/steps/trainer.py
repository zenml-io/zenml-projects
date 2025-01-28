import pytorch_lightning as pl
import torch.nn as nn
from lightning.pytorch.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from steps.model import LitResnet
from torch.utils.data import DataLoader
from typing_extensions import Annotated

from zenml import log_metadata, step
from zenml.artifacts.artifact_config import ArtifactConfig
from zenml.client import Client
from zenml.enums import ArtifactType
from zenml.integrations.neptune.experiment_trackers import (
    NeptuneExperimentTracker,
)
from zenml.integrations.neptune.experiment_trackers.run_state import (
    get_neptune_run,
)


@step(experiment_tracker="neptune_experiment_tracker", enable_cache=True)
def train_model(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int = 5,
    lr: float = 0.05,
) -> Annotated[nn.Module, ArtifactConfig(name="resnet18_model", artifact_type=ArtifactType.MODEL)]:
    """Train the model using PyTorch Lightning.
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Trained PyTorch model
    """
    # Initialize model
    model = LitResnet(lr=lr)
    
    # Get Neptune run and logger
    neptune_run = get_neptune_run()
    neptune_logger = NeptuneLogger(run=neptune_run)
    
    # Log hyperparameters to ZenML
    log_metadata(
        metadata={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": train_dataloader.batch_size,
            "optimizer": "SGD",
            "scheduler": "OneCycleLR",
        },
    )
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        enable_progress_bar=True,
        logger=neptune_logger,
    )
    
    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    return model.model  # Return the PyTorch model inside the Lightning module 