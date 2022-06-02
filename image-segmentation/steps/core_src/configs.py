from zenml.steps import BaseStepConfig


class PreTrainingConfigs(BaseStepConfig):
    """
    TODO:
    """

    seed = 101
    debug = False  # set debug=False for Full Training
    exp_name = "Baselinev2"
    comment = "unet-efficientnet_b1-224x224-aug2-split2"
    model_name = "Unet"
    backbone = "efficientnet-b1"
    train_bs = 128
    valid_bs = train_bs * 2
    img_size = [224, 224]
    epochs = 15
    lr = 2e-3
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 32 // train_bs)
    n_fold = 5
    num_classes = 3
    device = "cpu"
