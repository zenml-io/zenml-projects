import segmentation_models_pytorch as smp
import torch

from ..configs import PreTrainingConfigs


class ImageSegModel:
    def __init__(self) -> None:
        self.config = PreTrainingConfigs

    def build_model(self):
        model = smp.Unet(
            encoder_name=self.config.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.config.num_classes,  # model output channels (number of classes in your dataset)
            activation=None,
        )
        model.to(self.config.device)
        return model

    def load_model(self, path):
        model = self.build_model()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
