from typing import Annotated, List, Tuple

from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

from materializers.label_studio_yolo_dataset_materializer import (
    LabelStudioYOLODataset,
    LabelStudioYOLODatasetMaterializer,
)

logger = get_logger(__name__)


@step(
    output_materializers={"YOLO_dataset": LabelStudioYOLODatasetMaterializer}
)
def load_data_from_label_studio(
    dataset_name: str,
) -> Tuple[
    Annotated[LabelStudioYOLODataset, "YOLO_dataset"],
    Annotated[List[int], "new_ids"],
]:
    """Loads data from Label Studio.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Tuple of the loaded dataset and the Label Studio task IDs.
    """
    annotator = Client().active_stack.annotator
    from zenml.integrations.label_studio.annotators.label_studio_annotator import (
        LabelStudioAnnotator,
    )

    if not isinstance(annotator, LabelStudioAnnotator):
        raise TypeError(
            "This step can only be used with the Label Studio annotator."
        )

    if annotator and annotator._connection_available():
        try:
            dataset = annotator.get_dataset(dataset_name=dataset_name)
            ls_dataset = LabelStudioYOLODataset()
            ls_dataset.dataset = dataset

            c = Client()
            step_context = get_step_context()
            cur_pipeline_name = step_context.pipeline.name
            cur_step_name = step_context.step_name

            try:
                last_run = c.get_pipeline(
                    cur_pipeline_name
                ).last_successful_run
                last_task_ids = (
                    last_run.steps[cur_step_name].outputs["new_ids"].load()
                )
            except (RuntimeError, KeyError):
                last_task_ids = []

            current_labeled_task_ids = dataset.get_labeled_tasks_ids()
            logger.info(f"{len(current_labeled_task_ids)} total labels found.")

            new_task_ids = list(
                set(current_labeled_task_ids) - set(last_task_ids)
            )
            logger.info(
                f"{len(new_task_ids)} new labels are being beamed "
                f"straight to you."
            )

            ls_dataset.task_ids = new_task_ids
            return ls_dataset, new_task_ids
        except:
            raise ValueError(
                f"Dataset {dataset_name} not found in Label Studio."
            )

    else:
        raise TypeError(
            "This step can only be used with an active Label Studio annotator."
        )
