import segmentation_models_pytorch as smp
import torch

JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)


class LossFunctions:
    def __init__(self) -> None:
        pass

    def dice_coef(self, y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred > thr).to(torch.float32)
        inter = (y_true * y_pred).sum(dim=dim)
        den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
        dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
        return dice

    def iou_coef(self, y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
        y_true = y_true.to(torch.float32)
        y_pred = (y_pred > thr).to(torch.float32)
        inter = (y_true * y_pred).sum(dim=dim)
        union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
        iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
        return iou

    def criterion(self, y_pred, y_true):
        return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)
