# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
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

from zenml import step
from datasets import load_dataset, Dataset
from typing_extensions import Annotated
from typing import List,Dict,Any,Tuple
from uuid import uuid4


@step
def load_data(full_dataset_name: str = "daishen/CALM-Data")->Annotated[Dataset,"calm_dataset"]:
    dataset = load_dataset(full_dataset_name)
    return dataset['train']

@step
def convert_to_conversation(dataset: Dataset, dev_ratio: float = 0.05)->Tuple[Annotated[List[Dict[str,Any]],"calm_dev_dataset"],Annotated[List[Dict[str,Any]],"calm_train_dataset"]]:
    dataset_prepared = []
    for instruction_,input_,output_ in zip(dataset.data['instruction'],dataset.data['input'],dataset.data['output']):
        conversations = [{"from": "human", "value": str(instruction_)+str(input_)},{"from": "assistant", "value": str(output_)}]
        item = {"id":str(uuid4()),"conversations": conversations}
        dataset_prepared.append(item)
    
    if dev_ratio<=1:
        dev_ratio = int(len(dataset_prepared)*dev_ratio)
    return dataset_prepared[:dev_ratio],dataset_prepared[dev_ratio:]
