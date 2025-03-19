# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import logger
from zenml import step


@step
def push_model_to_huggingface(
    model_dir: str,
    tokenizer_dir: str,
    repo_id: str,
) -> str:
    """
    Loads the saved model and tokenizer and pushes them to the Hugging Face Hub.
    """

    logger.log_process("Loading model and tokenizer from disk...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    commit_message = "Deploying model to Hugging Face Hub"
    logger.log_process(f"Pushing model to {repo_id}...")
    model.push_to_hub(repo_id, commit_message=commit_message)
    tokenizer.push_to_hub(repo_id, commit_message=commit_message)

    logger.log_success(f"Model and tokenizer successfully pushed to {repo_id}")
    return repo_id
