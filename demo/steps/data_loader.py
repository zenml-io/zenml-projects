import os
import numpy as np
import torch
import torchvision
from typing import Tuple, Annotated, List
from torch.utils.data import DataLoader, random_split, Subset
from zenml import step

# Constants
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2) if os.cpu_count() else 2

# Data normalization
cifar10_normalization = torchvision.transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization,
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    cifar10_normalization,
])

def get_subset_indices(total_size: int, fraction: float) -> List[int]:
    """Get random indices for subset of data.
    
    Args:
        total_size: Total size of the dataset
        fraction: Fraction of data to use
        
    Returns:
        List of indices for the subset
    """
    num_samples = int(total_size * fraction)
    indices = np.random.permutation(total_size)[:num_samples].tolist()
    return indices

@step
def load_cifar10_data(
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.2,
    dataset_fraction: float = 0.05  # Use only 20% of the data by default
) -> Tuple[
    Annotated[DataLoader, "train_dataloader"],
    Annotated[DataLoader, "val_dataloader"],
    Annotated[DataLoader, "test_dataloader"]
]:
    """Load and prepare CIFAR10 datasets.
    
    Args:
        batch_size: Batch size for the dataloaders
        val_split: Fraction of training data to use for validation
        dataset_fraction: Fraction of total dataset to use (for faster demo)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load full datasets
    dataset_train_full = torchvision.datasets.CIFAR10(PATH_DATASETS, train=True, download=True, transform=train_transforms)
    dataset_test_full = torchvision.datasets.CIFAR10(PATH_DATASETS, train=False, download=True, transform=test_transforms)
    
    # Get subset indices
    train_indices = get_subset_indices(len(dataset_train_full), dataset_fraction)
    test_indices = get_subset_indices(len(dataset_test_full), dataset_fraction)
    # Create subsets
    dataset_train = Subset(dataset_train_full, train_indices)
    dataset_test = Subset(dataset_test_full, test_indices)
    
    # Split training into train and validation
    train_length = int(len(dataset_train) * (1 - val_split))
    val_length = len(dataset_train) - train_length
    dataset_train, dataset_val = random_split(
        dataset_train, 
        [train_length, val_length],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset sizes:")
    print(f"Original training set: {len(dataset_train_full)} samples")
    print(f"Original test set: {len(dataset_test_full)} samples")
    print(f"After {dataset_fraction*100:.1f}% subset:")
    print(f"  Training: {len(dataset_train)} samples")
    print(f"  Validation: {len(dataset_val)} samples")
    print(f"  Test: {len(dataset_test)} samples")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_dataloader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    return train_dataloader, val_dataloader, test_dataloader 