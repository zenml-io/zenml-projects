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

from steps import (
    trainer,
    merge_and_push
)

from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def finetune_starcoder():
    """
    This pipeline finetunes the starcoder model.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.
    trainer_obj, tokenizer, output_peft_repo_id, train_dataset, eval_dataset = trainer()
    merge_and_push(peft_model_id=output_peft_repo_id)
