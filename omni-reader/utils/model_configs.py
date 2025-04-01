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
"""Model configuration utilities for OCR operations."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ModelConfig:
    """Configuration for OCR models."""

    name: str
    display: str
    provider: str
    prefix: str
    logo: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    default_confidence: float = 0.5


# --------- Model info ---------
def get_model_info(model_name: str) -> Tuple[str, str]:
    """Returns a tuple (display, prefix) for a given model name.

    Args:
        model_name: The name of the model

    Returns:
        A tuple (display, prefix)
    """
    if model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name]
        return config.display, config.prefix

    # Fallback: Generate display name and prefix from model name
    if "/" in model_name:
        model_part = model_name.split("/")[-1]

        if ":" in model_part:
            display = model_part.split(":")[0]
        else:
            display = model_part

        display = display.replace("-", " ").title()
    else:
        display = model_name.split("-")[0]
        if ":" in display:
            display = display.split(":")[0]
        display = display.title()

    prefix = display.lower().replace(" ", "_").replace("-", "_")

    return display, prefix


# ---------  models ---------
MODEL_CONFIGS = {
    "mistral/pixtral-12b-2409": ModelConfig(
        name="mistral/pixtral-12b-2409",
        display="Mistral Pixtral 12B",
        provider="mistral",
        prefix="pixtral_12b_2409",
        logo="mistralai.svg",
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        display="GPT-4o-mini",
        provider="openai",
        prefix="openai_gpt_4o_mini",
        logo="openai.svg",
    ),
    "gemma3:12b": ModelConfig(
        name="gemma3:12b",
        display="Gemma 3 12B",
        provider="ollama",
        prefix="gemma3_12b",
        logo="gemma.svg",
    ),
    "gemma3:27b": ModelConfig(
        name="gemma3:27b",
        display="Gemma 3 27B",
        provider="ollama",
        prefix="gemma3_27b",
        logo="gemma.svg",
    ),
    "llama3.2-vision:11b": ModelConfig(
        name="llama3.2-vision:11b",
        display="Llama 3.2 Vision 11B",
        provider="ollama",
        prefix="llama3_2_vision_11b",
        logo="ollama.svg",
    ),
    "granite3.2-vision": ModelConfig(
        name="granite3.2-vision",
        display="Granite 3.2 Vision",
        provider="ollama",
        prefix="granite3_2_vision",
        logo="ollama.svg",
    ),
    "llava:7b": ModelConfig(
        name="llava:7b",
        display="Llava 7B",
        provider="ollama",
        prefix="llava_7b",
        logo="ollama.svg",
    ),
    "moondream": ModelConfig(
        name="moondream",
        display="Moondream",
        provider="ollama",
        prefix="moondream_v",
        logo="moondream.svg",
    ),
    "minicpm-v": ModelConfig(
        name="minicpm-v",
        display="MiniCPM-V",
        provider="ollama",
        prefix="minicpm_v",
        logo="ollama.svg",
    ),
}


DEFAULT_MODEL = MODEL_CONFIGS["mistral/pixtral-12b-2409"]
