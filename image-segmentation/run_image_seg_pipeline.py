from pipelines.run_image_seg_pipeline import image_segmentation_pipeline
from steps.data_steps import (
    apply_augmentations,
    create_stratified_fold,
    prepare_dataloaders,
    prepare_df,
)


def run_img_seg_pipe():
    """
    TODO:
    """
    image_seg_pipe = image_segmentation_pipeline(
        prepare_df(), create_stratified_fold(), prepare_dataloaders()
    )
    image_seg_pipe.run()


if __name__ == "__main__":
    run_img_seg_pipe()
