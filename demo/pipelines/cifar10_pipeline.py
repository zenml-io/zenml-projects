from typing import Dict

from steps.data_loader import load_cifar10_data
from steps.evaluator import evaluate_model
from steps.trainer import train_model

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.config.resource_settings import ResourceSettings
from zenml.integrations.constants import PYTORCH
from zenml.integrations.gcp.flavors.vertex_orchestrator_flavor import (
    VertexOrchestratorSettings,
)

vertex_settings = VertexOrchestratorSettings(
    pod_settings={
        "node_selectors": {
            "cloud.google.com/gke-accelerator": "NVIDIA_TESLA_V100",
        },
    }
)
#resource_settings = ResourceSettings(gpu_count=1)
resource_settings = ResourceSettings(cpu_count=16, memory="32GB")
@pipeline(
    settings={
        #"orchestrator": vertex_settings,
        "resources": resource_settings,
    },
    enable_cache=True
)
def cifar10_pipeline(
    batch_size: int = 256,
    val_split: float = 0.2,
    dataset_fraction: float = 0.05,  # Control dataset size
    epochs: int = 5,
    learning_rate: float = 0.05
) -> Dict[str, float]:
    """Training pipeline for CIFAR10 image classification.
    
    Args:
        batch_size: The batch size for training and evaluation.
        val_split: The fraction of the dataset to use for validation.
        dataset_fraction: The fraction of total dataset to use (for faster demo).
        epochs: The number of epochs to train the model.
        learning_rate: The learning rate for the optimizer.
    
    Returns:
        A dictionary containing the test loss and accuracy.
    """
    # Load and prepare data
    train_dataloader, val_dataloader, test_dataloader = load_cifar10_data(
        batch_size=batch_size,
        val_split=val_split,
        dataset_fraction=dataset_fraction
    )
    
    # Train model
    model = train_model(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=epochs,
        lr=learning_rate,
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_dataloader=test_dataloader
    )
    
    return metrics