import segmentation_models_pytorch as smp
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from zenml.integrations.constants import WANDB
from zenml.integrations.wandb.wandb_step_decorator import enable_wandb
from zenml.logger import get_logger
from zenml.repository import Repository
from zenml.steps import Output, step

from .core_src.configs import PreTrainingConfigs
from .core_src.model.build_model import ImageSegModel
from .core_src.model.run_training import TrainModel

logger = get_logger(__name__)

step_operator = Repository().active_stack.step_operator


@step
def initiate_model_and_optimizer(
    config: PreTrainingConfigs,
) -> Output(model=smp.Unet, optimizer=optim.Adam, sch=lr_scheduler.CosineAnnealingLR,):
    """
    It initializes the U-Net model, Optimizer, and Scheduler.

    Args:
        model: U-Net Image Segmentation model
        optimizer: Adam optimizer from torch
        scheduler: It fetches the scheduler
    """
    try:
        image_seg_model = ImageSegModel(config)
        models, optimizers, schedulers = image_seg_model.initiate_model_and_optimizer()
        return models, optimizers, schedulers
    except Exception as e:
        logger.error(e)
        raise e


@enable_wandb
@step(custom_step_operator=step_operator.name)
def train_model(
    model: smp.Unet,
    optimizer: optim.Adam,
    schedule: lr_scheduler.CosineAnnealingLR,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config: PreTrainingConfigs,
) -> Output(unet_model=smp.Unet, history=list):
    """
    `train_model` trains a model using the `TrainModel` class.

    Args:
        model: smp.Unet - The model to train
        optimizer: optim.Adam
        schedule: lr_scheduler.CosineAnnealingLR
        train_loader: DataLoader
        valid_loader: DataLoader
        config: PreTrainingConfigs
    """

    try:
        train_model = TrainModel()
        unet_model, history = train_model.run_training(
            0, model, optimizer, schedule, 15, train_loader, valid_loader, config
        )
        return unet_model, history
    except Exception as e:
        logger.error(e)
        raise e
