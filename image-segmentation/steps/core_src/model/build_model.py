from typing import Union

import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from zenml.steps import Output

from ..configs import PreTrainingConfigs


class ImageSegModel:
    """Image Segmentation Model"""

    def __init__(self, config: PreTrainingConfigs) -> None:
        """Initializes the class."""
        self.config = config

    def initiate_model_and_optimizer(
        self,
    ) -> Output(
        model=smp.Unet,
        optimizer=optim.Adam,
        scheduler=Union[
            lr_scheduler.CosineAnnealingLR,
            lr_scheduler.CosineAnnealingWarmRestarts,
            lr_scheduler.ReduceLROnPlateau,
            lr_scheduler.ExponentialLR,
        ],
    ):
        """
        It creates a model, optimizer and scheduler.
        """
        model = self.build_model()
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = self.fetch_scheduler(optimizer)
        return model, optimizer, scheduler

    def build_model(self):
        """
        `model = smp.Unet(encoder_name=self.config.backbone, encoder_weights="imagenet", in_channels=3,
        classes=self.config.num_classes, activation=None)`

        The `smp.Unet` function is a function from the `segmentation_models_pytorch` library. It takes in a
        number of arguments, which are explained below:

        - `encoder_name`: This is the name of the encoder that you want to use. In this case, we're using
        the `efficientnet-b7` encoder.
        - `encoder_weights`: This is the pre-trained weights that you want to use for the encoder. In this
        case, we're using the `imagenet` weights.
        - `in_channels`: This is the number of channels in the input image.
        """
        model = smp.Unet(
            encoder_name=self.config.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.config.num_classes,  # model output channels (number of classes in your dataset)
            activation=None,
        )
        model.to(self.config.device)
        return model

    def load_model(self, path: str):
        """
        It loads the model from the path and returns the model.

        Args:
            path: the path to the model you want to load
        """
        model = self.build_model()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

    def fetch_scheduler(
        self, optimizer: optim.Adam
    ) -> Output(
        scheduler=Union[
            lr_scheduler.CosineAnnealingLR,
            lr_scheduler.CosineAnnealingWarmRestarts,
            lr_scheduler.ExponentialLR,
        ]
    ):
        """
        It returns a scheduler object based on the config.scheduler parameter.

        Args:
            optimizer: The optimizer to use
        """

        if self.config.scheduler == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.T_max, eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "CosineAnnealingWarmRestarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.config.T_0, eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=7,
                threshold=0.0001,
                min_lr=self.config.min_lr,
            )
        elif self.config.scheduer == "ExponentialLR":
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        elif self.config.scheduler == None:
            return None

        return scheduler
