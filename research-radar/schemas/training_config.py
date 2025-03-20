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

from typing import Literal

from pydantic import BaseModel, Field
from transformers import TrainingArguments


class TrainingConfig(BaseModel):
    """Training configuration for ModernBERT model."""

    output_dir: str
    learning_rate: float = Field(default=3e-5, gt=0)
    per_device_train_batch_size: int = Field(default=8, gt=0)
    per_device_eval_batch_size: int = Field(default=64, gt=0)
    num_train_epochs: int = Field(default=8, gt=0)
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial"
    ] = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    eval_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    label_smoothing_factor: float = Field(default=0.1, ge=0, le=1)
    logging_dir: str = "./logs"
    logging_strategy: Literal["no", "steps", "epoch"] = "epoch"
    model_config = {
        "extra": "allow",
        "protected_namespaces": (),
    }

    def to_training_arguments(self) -> TrainingArguments:
        return TrainingArguments(**self.model_dump())

    @classmethod
    def from_dict(cls, config: dict) -> "TrainingConfig":
        return cls(**config)
