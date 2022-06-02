from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def image_segmentation_pipeline(prepare_df, create_stratified_fold, prepare_dataloaders):
    """
    TODO
    """
    df = prepare_df()
    fold_dfs = create_stratified_fold(df)
    train_loader, valid_loader = prepare_dataloaders(fold_dfs)
