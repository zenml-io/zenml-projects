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

from pathlib import Path
from zenml import step, get_step_context
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_one_example(
    target_sentence: str = "Earlier, you stated that you didn't have "
    "strong feelings about PlayStation's Little Big Adventure. Is "
    "your opinion true for all games which don't have multiplayer?",
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base_model_id = get_step_context().model.load_artifact("base_model_id")
    get_step_context().model.load_artifact("ft_model")
    model_path = Path("model_dir")
    ft_model = AutoModelForCausalLM.from_pretrained(model_path)

    eval_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
        trust_remote_code=True,
    )

    eval_prompt = f"""{get_step_context().model.load_artifact("system_prompt")}

### Target sentence:
{target_sentence}

### Meaning representation:
"""

    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad():
        logger.info(
            eval_tokenizer.decode(
                ft_model.generate(**model_input, max_new_tokens=100)[0],
                skip_special_tokens=True,
            )
        )
