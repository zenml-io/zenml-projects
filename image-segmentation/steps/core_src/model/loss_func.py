import numpy as np
import segmentation_models_pytorch as smp
import torch

JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)


class LossFunctions:
    """Class for all Evaluation Metrics."""

    def __init__(self) -> None:
        pass

    def dice_coef(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thr: float = 0.5,
        dim: tuple = (2, 3),
        epsilon: float = 0.001,
    ):
        """
        The function takes in the ground truth and the predicted mask, and returns the dice coefficient for
        each class.

        y_true: the ground truth mask
        y_pred: the output of the model
        thr: threshold for the prediction
        dim: the dimensions to sum over. In this case, we're summing over the last two dimensions,
        which are the height and width of the image
        epsilon: A small value to avoid division by zero
        """
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred > thr).to(torch.float32)
        inter = (y_true * y_pred).sum(dim=dim)
        den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
        dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
        return dice

    def iou_coef(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thr: float = 0.5,
        dim: tuple = (2, 3),
        epsilon: float = 0.001,
    ):
        """
        It calculates the IoU for each image in the batch.

        y_true: the ground truth mask
        y_pred: the output of the model
        thr: the threshold for the prediction to be considered positive
        dim: the dimensions to calculate the IoU over
        epsilon: A small value to avoid division by zero
        """
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred > thr).to(torch.float32)
        inter = (y_true * y_pred).sum(dim=dim)
        union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
        iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
        return iou

    def criterion(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        The function takes in two arguments, `y_pred` and `y_true`, and returns the sum of the binary
        cross entropy loss and the Tversky loss

        y_pred: The predicted output of the model
        y_true: the ground truth mask
        """
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)
