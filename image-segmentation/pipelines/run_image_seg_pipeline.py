from asyncio.windows_utils import pipe

from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def image_segmentation_pipeline(prepare_df, create_stratified_fold):
    pass
