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

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import instructor
from dotenv import load_dotenv
from litellm import completion
from mistralai import Mistral
from openai import OpenAI
from zenml.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for OCR models."""

    name: str
    client_factory: Callable
    prefix: str
    display: str = ""
    default_confidence: float = 0.5
    max_tokens: Optional[int] = None
    additional_params: Dict[str, Any] = None


def get_openai_client():
    """Get an OpenAI client with instructor integration."""
    openai_client = OpenAI(api_key="ollama")
    return instructor.from_openai(openai_client)


def get_mistral_client():
    """Get a Mistral client with instructor integration."""
    mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    return instructor.from_mistral(mistral_client)


def get_gemma_client():
    """Get a Gemma client with instructor integration."""
    return instructor.from_litellm(completion)


MODEL_CONFIGS = {
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        client_factory=get_openai_client,
        prefix="openai_gpt_4o_mini",
        default_confidence=0.75,
        max_tokens=1500,
        display="GPT-4o Mini",
    ),
    "ollama/gemma3:27b": ModelConfig(
        name="ollama/gemma3:27b",
        client_factory=get_gemma_client,
        prefix="ollama_gemma3_27b",
        display="Gemma-3-27B",
    ),
    "ollama/gemma3:12b": ModelConfig(
        name="ollama/gemma3:12b",
        client_factory=get_gemma_client,
        prefix="ollama_gemma3_12b",
        display="Gemma-3-12B",
    ),
    "ollama/gemma3:4b": ModelConfig(
        name="ollama/gemma3:4b",
        client_factory=get_gemma_client,
        prefix="ollama_gemma3_4b",
        display="Gemma-3-4B",
    ),
    "pixtral-12b-2409": ModelConfig(
        name="pixtral-12b-2409",
        client_factory=get_mistral_client,
        prefix="pixtral_12b_2409",
        display="Pixtral-12B-2409",
    ),
}


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
