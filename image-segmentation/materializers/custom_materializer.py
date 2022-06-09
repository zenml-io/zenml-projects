import os
import pickle
from typing import Any, Type, Union

import albumentations as A
import torch.optim as optim
from torch.optim import lr_scheduler
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME = "ImageSegmentationEnvironment"
import segmentation_models_pytorch as smp


class ImageCustomerMaterializer(BaseMaterializer):
    """
    Custom materializer for the Image Segmentation ZenFile
    """

    ASSOCIATED_TYPES = [optim.Adam, lr_scheduler.CosineAnnealingLR, smp.Unet, A.Compose, dict]

    def handle_input(
        self, data_type: Type[Any]
    ) -> Union[optim.Adam, lr_scheduler.CosineAnnealingLR, smp.Unet, A.Compose, dict]:
        """
        It loads the model from the artifact and returns it.

        Args:
            data_type: The type of the model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def handle_return(
        self, obj: Union[optim.Adam, lr_scheduler.CosineAnnealingLR, smp.Unet, A.Compose, dict]
    ) -> None:
        """
        It saves the model to the artifact store.

        Args:
            model: The model to be saved
        """

        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(obj, fid)
