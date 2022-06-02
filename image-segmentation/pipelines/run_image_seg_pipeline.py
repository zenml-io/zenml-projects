from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def image_segmentation_pipeline(
    prepare_df, create_stratified_fold, apply_augmentations, prepare_dataloaders
):
    """
    TODO
    """
    df = prepare_df()
    fold_dfs = create_stratified_fold(df)
    data_transforms = apply_augmentations()
    train_loader, valid_loader = prepare_dataloaders(fold_dfs)
