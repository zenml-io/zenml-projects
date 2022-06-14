import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

tqdm.pandas()

import gc

import numpy as np

from ..configs import PreTrainingConfigs
from .loss_func import LossFunctions

loss_fn = LossFunctions()
from typing import Union

from zenml.steps import Output


class TrainValModel:
    """This class is used to train the model."""

    def __init__(self) -> None:
        """Initialize the class."""
        pass

    def train_one_epoch(
        self,
        model: smp.Unet,
        optimizer: optim.Adam,
        scheduler: Union[
            lr_scheduler.CosineAnnealingLR,
            lr_scheduler.CosineAnnealingWarmRestarts,
            lr_scheduler.ReduceLROnPlateau,
            lr_scheduler.ExponentialLR,
        ],
        dataloader: DataLoader,
        device: str,
        config: PreTrainingConfigs,
    ) -> float:
        """
        Run Training for one epoch, and return the training loss.
        """

        model.train()
        scaler = amp.GradScaler()

        dataset_size = 0
        running_loss = 0.0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train ")
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            batch_size = images.size(0)

            with amp.autocast(enabled=True):
                y_pred = model(images)
                loss = loss_fn.criterion(y_pred, masks)
                loss = loss / config.n_accumulate

            scaler.scale(loss).backward()

            if (step + 1) % config.n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()

                # zero the parameter gradients
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                train_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_mem=f"{mem:0.2f} GB"
            )
        torch.cuda.empty_cache()
        gc.collect()

        return epoch_loss

    @torch.no_grad()
    def valid_one_epoch(
        self, model: smp.Unet, optimizer: optim.Adam, dataloader: DataLoader, device: str
    ) -> Output(epoch_loss=float, val_scores=float):
        """
        It validates the model per epoch.
        """
        model.eval()

        dataset_size = 0
        running_loss = 0.0

        val_scores = []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid")
        print(dataloader)
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            batch_size = images.size(0)

            y_pred = model(images)
            loss = loss_fn.criterion(y_pred, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            y_pred = nn.Sigmoid()(y_pred)
            val_dice = loss_fn.dice_coef(masks, y_pred).cpu().detach().numpy()
            val_jaccard = loss_fn.iou_coef(masks, y_pred).cpu().detach().numpy()
            val_scores.append([val_dice, val_jaccard])

            mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                valid_loss=f"{epoch_loss:0.4f}",
                lr=f"{current_lr:0.5f}",
                gpu_memory=f"{mem:0.2f} GB",
            )
        val_scores = np.mean(val_scores, axis=0)
        torch.cuda.empty_cache()
        gc.collect()
        return val_scores
