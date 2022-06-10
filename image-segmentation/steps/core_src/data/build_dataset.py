import numpy as np
import pandas as pd
import torch

from ..utils_classes import ImageUtils

image_utils = ImageUtils()
from zenml.steps import Output


class BuildDataset(torch.utils.data.Dataset):
    """
    This class takes in a Dataframe, and returns a dataset object that can be used to train a model.
    """

    def __init__(self, df: pd.DataFrame, label: bool = True, transforms=None) -> None:
        """Initializes the class."""
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.msk_paths = df["mask_path"].tolist()
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.df)

    def __getitem__(self, index: int) -> Output(image=torch.Tensor, mask=torch.Tensor):
        """Returns the item at the given index."""
        img_path = self.img_paths[index]
        img = []
        img = image_utils.load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = image_utils.load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data["image"]
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)
