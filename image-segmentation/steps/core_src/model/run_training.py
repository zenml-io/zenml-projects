import copy
import gc
import time
from collections import defaultdict
from typing import Union

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .train_val_model import TrainValModel

train_val_obj = TrainValModel()
from zenml.logger import get_logger

from ..configs import PreTrainingConfigs

logger = get_logger(__name__)

from zenml.steps import Output


class TrainModel:
    """This class is used to train the model."""

    def __init__(self) -> None:
        pass

    def run_training(
        self,
        fold: int,
        model: smp.Unet,
        optimizer: optim.Adam,
        scheduler: Union[
            lr_scheduler.CosineAnnealingLR,
            lr_scheduler.CosineAnnealingWarmRestarts,
            lr_scheduler.ReduceLROnPlateau,
            lr_scheduler.ExponentialLR,
        ],
        num_epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: PreTrainingConfigs,
    ) -> Output(model=smp.Unet, history=list):
        """
        It trains the model for a given number of epochs, and returns the best model and the training
        history.

        fold: The fold number
        model: The model to train
        optimizer: The optimizer used to train the model
        scheduler: The scheduler object that will be used to adjust the learning rate
        device: The device to run the training on
        num_epochs: Number of epochs to train for
        train_loader: The training data loader
        valid_loader: The validation data loader
        config: PreTrainingConfigs
        """

        # To automatically log gradients
        wandb.watch(model, log_freq=100)

        if torch.cuda.is_available():
            print("cuda: {}\n".format(torch.cuda.get_device_name()))

        start = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_dice = -np.inf
        best_epoch = -1
        history = defaultdict(list)
        for epoch in range(1, num_epochs + 1):
            gc.collect()
            print(f"Epoch {epoch}/{num_epochs}", end="")
            train_loss = train_val_obj.train_one_epoch(
                model,
                optimizer,
                scheduler,
                dataloader=train_loader,
                device=config.device,
                config=config,
            )

            val_scores = train_val_obj.valid_one_epoch(
                model, optimizer, valid_loader, device=config.device
            )
            val_dice = val_scores

            history["Train Loss"].append(train_loss)
            history["Valid Dice"].append(val_dice)

            # Log the metrics
            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Valid Dice": val_dice,
                    "LR": scheduler.get_last_lr()[0],
                }
            )

            logger.info(f"Valid Dice: {val_dice:0.4f}")

            # deep copy the model
            if val_dice >= best_dice:
                print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
                best_dice = val_dice
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"best_epoch-{fold:02d}.bin"
                torch.save(model.state_dict(), PATH)
                # Save a model file from the current directory
                wandb.save(PATH)
                logger.info(f"Model Saved")

            last_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"last_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            logger.info(f"Model Saved")

        end = time.time()
        time_elapsed = end - start
        logger.info(
            "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60
            )
        )
        # logger.info("Best Score: {:.4f}".format(best_jaccard))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, history
