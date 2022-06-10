import pandas as pd
from torch.utils.data import DataLoader

from ..configs import PreTrainingConfigs
from .build_dataset import BuildDataset
from .data_process import ProcessData

process_data = ProcessData()
from zenml.steps import Output


class CustomDataLoader:
    """Custom Data Loader that uses PyTorch DataLoader Module to load data."""

    def __init__(
        self, fold: int, df: pd.DataFrame, data_transform: dict, config: PreTrainingConfigs
    ) -> None:
        """Initializes the class."""
        self.fold = fold
        self.df = df
        self.data_transform = data_transform
        self.config = config

    def apply_loaders(self) -> Output(train_loader=DataLoader, valid_loader=DataLoader):
        """It prepares the data loaders and returns the train and valid loaders."""
        train_loader, valid_loader = self.prepare_loaders(self.fold, self.config, debug=True)
        return train_loader, valid_loader

    def prepare_loaders(
        self, fold: int, config: PreTrainingConfigs, debug: bool = False
    ) -> Output(train_loader=DataLoader, valid_loader=DataLoader):
        """It prepares the data loaders and returns the train and valid loaders.

        Args:
            fold: int
            config: PreTrainingConfigs
            debug: bool
        """
        train_df = self.df.query("fold!=@fold").reset_index(drop=True)
        valid_df = self.df.query("fold==@fold").reset_index(drop=True)
        if debug:
            train_df = train_df.head(32 * 5).query("empty==0")
            valid_df = valid_df.head(32 * 3).query("empty==0")
        train_dataset = BuildDataset(train_df, transforms=self.data_transform["train"])
        valid_dataset = BuildDataset(valid_df, transforms=self.data_transform["valid"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_bs if not debug else 20,
            num_workers=4,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.valid_bs if not debug else 20,
            num_workers=4,
            shuffle=False,
            pin_memory=True,
        )

        return train_loader, valid_loader
