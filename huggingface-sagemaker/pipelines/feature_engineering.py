# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Optional

from zenml import pipeline
from zenml.logger import get_logger

from steps import (
    data_loader,
    notify_on_failure,
    tokenization_step,
    tokenizer_loader,
    generate_reference_and_comparison_datasets,
)
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps import (
    EvidentlyColumnMapping,
    evidently_report_step,
)

logger = get_logger(__name__)


@pipeline(on_failure=notify_on_failure)
def sentinment_analysis_feature_engineering_pipeline(
    lower_case: Optional[bool] = True,
    padding: Optional[str] = "max_length",
    max_seq_length: Optional[int] = 128,
    text_column: Optional[str] = "text",
    label_column: Optional[str] = "label",
):
    """
    Model training pipeline.

    This is a pipeline that loads the datataset and tokenzier,
    tokenizes the dataset, trains a model and registers the model
    to the model registry.

    Args:
        lower_case: Whether to convert all text to lower case.
        padding: Padding strategy.
        max_seq_length: Maximum sequence length.
        text_column: Name of the text column.
        label_column: Name of the label column.
        train_batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.
        num_epochs: Number of epochs.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.

    ########## Load Dataset stage ##########
    dataset = data_loader()

    ########## Data Quality stage ##########
    reference_dataset, comparison_dataset = generate_reference_and_comparison_datasets(
        dataset
    )
    text_data_report = evidently_report_step.with_options(
        parameters=dict(
            column_mapping=EvidentlyColumnMapping(
                target="label",
                text_features=["text"],
            ),
            metrics=[
                EvidentlyMetricConfig.metric("DataQualityPreset"),
                EvidentlyMetricConfig.metric(
                    "TextOverviewPreset", column_name="text"
                ),
            ],
            # We need to download the NLTK data for the TextOverviewPreset
            download_nltk_data=True,
        ),
    )
    text_data_report(reference_dataset, comparison_dataset)

    ########## Tokenization stage ##########
    tokenizer = tokenizer_loader(lower_case=lower_case)
    tokenized_data = tokenization_step(
        dataset=dataset,
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        text_column=text_column,
        label_column=label_column,
    )
    return tokenizer, tokenized_data
