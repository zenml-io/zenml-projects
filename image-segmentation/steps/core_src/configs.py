from zenml.steps import BaseStepConfig


class PreTrainingConfigs(BaseStepConfig):
    """
    Pre-Training Configuration Class.
    """

    seed: int = 101
    debug: bool = False  # set debug=False for Full Training
    exp_name: str = "Baselinev2"
    comment: str = "unet-efficientnet_b1-224x224-aug2-split2"
    model_name: str = "Unet"
    backbone: str = "efficientnet-b1"
    train_bs: int = 128
    valid_bs: int = train_bs * 2
    img_size: list = [224, 224]
    epochs: int = 15
    lr: float = 2e-3
    scheduler: str = "CosineAnnealingLR"
    min_lr: float = 1e-6
    T_max: float = int(30000 / train_bs * epochs) + 50
    T_0: int = 25
    warmup_epochs: int = 0
    wd: float = 1e-6
    n_accumulate: float = max(1, 32 // train_bs)
    n_fold: int = 5
    num_classes: int = 3
    device: str = "cpu"
