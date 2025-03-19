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

from pathlib import Path
from typing import Annotated, Tuple

from schemas import zenml_project
from transformers import (
    ModernBertForSequenceClassification,
    PreTrainedTokenizerFast,
)
from utils import logger
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType


@step(model=zenml_project)
def save_model_local(
    model: ModernBertForSequenceClassification,
    tokenizer: PreTrainedTokenizerFast,
    model_dir: str,
    tokenizer_dir: str,
) -> Tuple[
    Annotated[
        str,
        ArtifactConfig(
            name="ft_model_dir",
            artifact_type=ArtifactType.MODEL,
            tags=["exported", "local"],
        ),
    ],
    Annotated[
        str,
        ArtifactConfig(
            name="ft_tokenizer_dir",
            artifact_type=ArtifactType.MODEL,
            tags=["exported", "local"],
        ),
    ],
]:
    """Exports model and tokenizer artifacts locally.

    Args:
        model: Fine-tuned ModernBERT model
        tokenizer: Associated tokenizer
        model_dir: Directory to save model
        tokenizer_dir: Directory to save tokenizer

    Returns:
        Tuple[str, str]: Paths to saved model and tokenizer
    """
    output_model_dir = Path(model_dir)
    output_tokenizer_dir = Path(tokenizer_dir)

    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_tokenizer_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    logger.log_success(f"Model saved to {model_dir}")
    logger.log_success(f"Tokenizer saved to {tokenizer_dir}")

    return str(output_model_dir), str(output_tokenizer_dir)
