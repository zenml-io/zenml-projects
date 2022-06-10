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
    """We have a pipeline which prepares the dataframe, create stratified folds, augment the dataframe, prepare the dataloaders,
    initiate the model and optimizer, and train the model

    Args:
        prepare_df: This step will read the data and prepare it for the pipeline.
        create_stratified_fold: This step creates stratified k folds.
        augment_df: This is a step that returns a dictionary of data transforms.
        prepare_dataloaders: This step takes in the dataframe and the data transforms and returns
        the train and validation dataloaders.
        initiate_model_and_optimizer: This is a step that returns a tuple of (model, optimizer,
        scheduler).
        train_model: a step that takes in the model, optimizer, scheduler, train_loader, and
        valid_loader and returns the trained model and history.
    """
    df = prepare_df()
    fold_dfs = create_stratified_fold(df)
    data_transforms = augment_df()
    train_loader, valid_loader = prepare_dataloaders(fold_dfs, data_transforms)
    models, optimizers, schedulers = initiate_model_and_optimizer()
    model, history = train_model(models, optimizers, schedulers, train_loader, valid_loader)
