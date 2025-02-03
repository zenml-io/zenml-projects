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

from steps.finetune_embeddings import (
    evaluate_base_model,
    evaluate_finetuned_model,
    finetune,
    prepare_load_data,
    visualize_results,
)
from zenml import pipeline


@pipeline
def finetune_embeddings():
    data = prepare_load_data()
    base_results = evaluate_base_model(dataset=data)
    finetune(dataset=data)
    finetuned_results = evaluate_finetuned_model(
        dataset=data, after="finetune"
    )
    visualize_results(base_results, finetuned_results)


if __name__ == "__main__":
    finetune_embeddings()
