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

from typing import List, Optional
from pydantic import BaseModel
from zenml.utils.enum_utils import StrEnum


class Configuration(BaseModel):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_path: str = "bigcode/starcoderplus"
    dataset_name: str = "smangrul/hf-stack-v1"
    subset: str = "data"
    split: str = "train"
    size_valid_set: int = 4000
    test_size: float = 0.005
    streaming: bool = False
    shuffle_buffer: int = 5000
    data_column: str = "content"

    seq_length: int = 8192
    max_steps: int = 10000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    eos_token_id: int = 49152
    num_train_epochs: int = 3

    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    weight_decay: float = 0.05

    local_rank: int = 0
    no_fp16: bool = True
    bf16: bool = False
    no_gradient_checkpointing: bool = True
    seed: int = 0
    num_workers: Optional[int] = None
    output_dir: str = "./checkpoints"
    log_freq: int = 1
    eval_freq: int = 1000
    save_freq: int = 1000

    fim_rate: float = 0
    fim_spm_rate: float = 0

    use_peft_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = [
        "q_proj",
        "o_proj",
        "v_proj",
        "gate_proj",
        "down_proj",
        "k_proj",
        "up_proj",
    ]

    use_flash_attn: bool = False

    use_4bit_qunatization: bool = False
    use_nested_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"

    use_8bit_qunatization: bool = False

    push_to_hub: bool = False
    output_peft_repo_id: str = "htahir1/peft-lora-zencoder15B-personal-copilot"

    should_log: bool = True
    try_resume_from_checkpoint: bool = True
