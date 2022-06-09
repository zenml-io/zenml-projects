from zenml.integrations.constants import WANDB
from zenml.pipelines import pipeline


@pipeline(enable_cache=True, required_integrations=[WANDB])
def image_segmentation_pipeline(
    prepare_df,
    create_stratified_fold,
    augment_df,
    prepare_dataloaders,
    initiate_model_and_optimizer,
    train_model,
):
    """
    TODO
    """
    df = prepare_df()
    fold_dfs = create_stratified_fold(df)
    data_transforms = augment_df()
    train_loader, valid_loader = prepare_dataloaders(fold_dfs, data_transforms)
    models, optimizers, schedulers = initiate_model_and_optimizer()
    model, history = train_model(models, optimizers, schedulers, train_loader, valid_loader)
