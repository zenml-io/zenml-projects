from materializers.custom_materializer import ImageCustomerMaterializer
from pipelines.run_image_seg_pipeline import image_segmentation_pipeline
from steps.data_steps import (
    apply_augmentations,
    create_stratified_fold,
    prepare_dataloaders,
    prepare_df,
)
from steps.model_steps import initiate_model_and_optimizer, train_model


def run_img_seg_pipe():
    image_seg_pipe = image_segmentation_pipeline(
        prepare_df(),
        create_stratified_fold(),
        apply_augmentations().with_return_materializers(ImageCustomerMaterializer),
        prepare_dataloaders(),
        initiate_model_and_optimizer().with_return_materializers(ImageCustomerMaterializer),
        train_model(),
    )
    image_seg_pipe.run()


if __name__ == "__main__":
    run_img_seg_pipe()
