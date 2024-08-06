#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from constants import (
    DATASET_NAME_ARGILLA_EMBEDDINGS,
    DATASET_NAME_DISTILABEL_EMBEDDINGS,
    EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS,
    EMBEDDINGS_MODEL_NAME_BASELINE,
    EMBEDDINGS_MODEL_NAME_FINE_TUNED,
    EMBEDDINGS_MODEL_NAME_ZENML,
)
from steps.finetune_embeddings import (
    evaluate_base_model,
    evaluate_finetuned_model,
    finetune,
    prepare_load_data,
)
from zenml import Model, pipeline
from zenml.model.model import ModelStages

model_definition = Model(
    name=EMBEDDINGS_MODEL_NAME_ZENML,
    version=ModelStages.LATEST,
)


@pipeline(
    model=model_definition,
)
def finetune_embeddings():
    data = prepare_load_data(
        dataset_name_argilla=DATASET_NAME_ARGILLA_EMBEDDINGS,
        dataset_name_hf=DATASET_NAME_DISTILABEL_EMBEDDINGS,
    )
    evaluate_base_model(
        dataset=data,
        model_original=EMBEDDINGS_MODEL_NAME_BASELINE,
        matryoshka_dims=EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS
    )
    finetune(
        dataset=data,
        model_orginal=EMBEDDINGS_MODEL_NAME_BASELINE,
        model_fine_tuned=EMBEDDINGS_MODEL_NAME_FINE_TUNED,
        matryoshka_dims=EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS
    )
    evaluate_finetuned_model(
        dataset=data,
        model_fine_tuned=EMBEDDINGS_MODEL_NAME_FINE_TUNED,
        matryoshka_dims=EMBEDDINGS_MODEL_MATRYOSHKA_DIMENSIONS,
        after="finetune"
    )


if __name__ == "__main__":
    finetune_embeddings()
