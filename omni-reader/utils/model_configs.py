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
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import instructor
import requests
from mistralai import Mistral
from openai import OpenAI
from zenml.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for OCR models."""

    name: str
    display: str
    provider: str
    prefix: str
    logo: Optional[str] = None
    base_url: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    default_confidence: float = 0.5

    def get_client(self):
        """Get the appropriate client for this model configuration."""
        if self.provider == "openai":
            return get_openai_client()
        elif self.provider == "mistral":
            return get_mistral_client()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def process_image(self, prompt, image_base64, content_type="image/jpeg"):
        """Process an image with this model."""
        if self.provider == "ollama":
            return self._process_ollama(prompt, image_base64)
        else:
            return self._process_api_based(prompt, image_base64, content_type)

    def _process_ollama(self, prompt, image_base64):
        """Process an image with an Ollama model."""
        from utils.ocr_processing import try_extract_json_from_response

        base_url = self.base_url or DOCKER_BASE_URL

        payload = {
            "model": self.name,
            "prompt": prompt,
            "stream": False,
            "images": [image_base64],
        }

        try:
            response = requests.post(
                base_url,
                json=payload,
                timeout=120,  # Increase timeout for larger images
            )
            response.raise_for_status()
            res = response.json().get("response", "")
            result_json = try_extract_json_from_response(res)

            return result_json
        except Exception as e:
            logger.error(f"Error processing with Ollama model {self.name}: {str(e)}")
            return {"raw_text": f"Error: {str(e)}", "confidence": 0.0}

    def _process_api_based(self, prompt, image_base64, content_type):
        """Process an image with an API-based model (OpenAI, Mistral)."""
        from utils.ocr_processing import try_extract_json_from_response
        from utils.prompt import ImageDescription

        client = self.get_client()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{content_type};base64,{image_base64}"},
                    },
                ],
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.name,
                messages=messages,
                response_model=ImageDescription,
                **self.additional_params,
            )

            result_json = try_extract_json_from_response(response)
            return result_json
        except Exception as e:
            logger.error(f"Error processing with {self.provider} model {self.name}: {str(e)}")
            return {"raw_text": f"Error: {str(e)}", "confidence": 0.0}


# --------- Ollama models ---------
DOCKER_BASE_URL = "http://host.docker.internal:11434/api/generate"
BASE_URL = "http://localhost:11434/api/generate"


def get_openai_client():
    """Get an OpenAI client with instructor integration."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return instructor.from_openai(client)


def get_mistral_client():
    """Get a Mistral client with instructor integration."""
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    return instructor.from_mistral(client)


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
    "pixtral-12b-2409": ModelConfig(
        name="pixtral-12b-2409",
        display="Mistral Pixtral 12B",
        provider="mistral",
        prefix="pixtral_12b_2409",
        logo="mistral.svg",
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
        base_url=BASE_URL,
    ),
    "llama3.2-vision:11b": ModelConfig(
        name="llama3.2-vision:11b",
        display="Llama 3.2 Vision 11B",
        provider="ollama",
        prefix="llama3_2_vision_11b",
        logo="ollama.svg",
        base_url=BASE_URL,
    ),
    "granite3.2-vision": ModelConfig(
        name="granite3.2-vision",
        display="Granite 3.2 Vision",
        provider="ollama",
        prefix="granite3_2_vision",
        logo="granite.svg",
        base_url=BASE_URL,
    ),
    "llava:7b": ModelConfig(
        name="llava:7b",
        display="Llava 7B",
        provider="ollama",
        prefix="llava_7b",
        logo="llava.svg",
        base_url=BASE_URL,
    ),
    "moondream": ModelConfig(
        name="moondream",
        display="Moondream",
        provider="ollama",
        prefix="moondream_v",
        logo="moondream.svg",
        base_url=BASE_URL,
    ),
    "minicpm-v": ModelConfig(
        name="minicpm-v",
        display="MiniCPM-V",
        provider="ollama",
        prefix="minicpm_v",
        logo="ollama.svg",
        base_url=BASE_URL,
    ),
    "qwen2:latest": ModelConfig(
        name="qwen2:latest",
        display="Qwen2",
        provider="ollama",
        prefix="qwen2_latest",
        logo="qwen.svg",
        base_url=BASE_URL,
    ),
}

DEFAULT_MODEL = MODEL_CONFIGS["llama3.2-vision:11b"]
