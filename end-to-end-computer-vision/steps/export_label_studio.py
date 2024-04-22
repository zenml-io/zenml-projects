from typing import Annotated, Tuple, List

from zenml import step, get_step_context
from zenml.client import Client

from materializers.label_studio_yolo_dataset_materializer import \
    LabelStudioYOLODatasetMaterializer, LabelStudioYOLODataset


@step(output_materializers={"yolo_dataset": LabelStudioYOLODatasetMaterializer})
def load_data_from_label_studio(
    dataset_name: str
) -> Tuple[
    Annotated[LabelStudioYOLODataset, "yolo_dataset"],
    Annotated[List[int], "task_ids"]
]:
    annotator = Client().active_stack.annotator
    from zenml.integrations.label_studio.annotators.label_studio_annotator import (
        LabelStudioAnnotator,
    )

    if not isinstance(annotator, LabelStudioAnnotator):
        raise TypeError(
            "This step can only be used with the Label Studio annotator."
        )

    if annotator and annotator._connection_available():
        for dataset in annotator.get_datasets():
            if dataset.get_params()["title"] == dataset_name:
                ls_dataset = LabelStudioYOLODataset()
                ls_dataset.dataset = dataset

                c = Client()
                step_context = get_step_context()
                cur_pipeline_name = step_context.pipeline.name
                cur_step_name = step_context.step_name

                try:
                    last_run = c.get_pipeline(cur_pipeline_name).last_successful_run
                    last_task_ids = last_run.steps[cur_step_name].outputs[
                        'task_ids'].read()
                except (RuntimeError, KeyError):
                    last_task_ids = []

                cur_task_ids = dataset.get_tasks_ids()
                new_task_ids = list(set(cur_task_ids) - set(last_task_ids))
                ls_dataset.task_ids = new_task_ids
                return ls_dataset, new_task_ids
    else:
        raise TypeError(
            "This step can only be used with an active Label Studio annotator."
        )
