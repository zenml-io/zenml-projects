from typing import Union

import segmentation_models_pytorch as smp
import torch.optim as optim
from src.configs import PreTrainingConfigs
from src.model.build_model import ImageSegModel
from torch.optim import lr_scheduler
from zenml.logger import get_logger
from zenml.steps import Output, step

logger = get_logger(__name__)


@step
def initiate_model_and_optimizer(
    config: PreTrainingConfigs,
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
    TODO:
    """
    try:
        image_seg_model = ImageSegModel(config)
        model, optimizer, scheduler = image_seg_model.initiate_model_and_optimizer()
        return model, optimizer, scheduler
    except Exception as e:
        logger.error(e)
        raise e
