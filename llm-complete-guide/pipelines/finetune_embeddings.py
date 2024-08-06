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

from constants import EMBEDDINGS_MODEL_NAME_ZENML
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
    data = prepare_load_data()
    evaluate_base_model(dataset=data)
    finetune(dataset=data)
    evaluate_finetuned_model(dataset=data, after="finetune")

if __name__ == "__main__":
    finetune_embeddings()
