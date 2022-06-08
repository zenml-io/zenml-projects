from zenml.integrations.constants import WANDB
from zenml.pipelines import pipeline


@pipeline(enable_cache=True, required_integrations=[WANDB])
def image_segmentation_pipeline(
    prepare_df,
    create_stratified_fold,
    prepare_dataloaders,
    initiate_model_and_optimizer,
    train_model,
):
    """
    TODO
    """
    df = prepare_df()
    fold_dfs = create_stratified_fold(df)
    train_loader, valid_loader = prepare_dataloaders(fold_dfs)
    models, optimizers, schedulers = initiate_model_and_optimizer()
    model, history = train_model(models, optimizers, schedulers, train_loader, valid_loader)
