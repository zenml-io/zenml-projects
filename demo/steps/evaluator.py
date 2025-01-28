from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zenml import log_metadata, step


@step(enable_cache=True)
def evaluate_model(
    model: nn.Module,
    test_dataloader: DataLoader,
) -> Dict[str, float]:
    """Evaluate the model on test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = float(test_loss) / len(test_dataloader.dataset)
    accuracy = 100.0 * float(correct) / len(test_dataloader.dataset)
    
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": accuracy
    }
    
    # Log test metrics to ZenML
    log_metadata(
        metadata={
            "test/loss": test_loss,
            "test/accuracy": accuracy,
            "test/total_samples": len(test_dataloader.dataset),
            "test/correct_samples": int(correct)
        },
    )
    
    print(f"Test set: Average loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_dataloader.dataset)} "
          f"({accuracy:.2f}%)")
    
    return metrics 