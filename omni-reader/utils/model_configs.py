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

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from utils.config import load_config


@dataclass
class ModelConfig:
    """Configuration for OCR models."""

    name: str
    ocr_processor: str
    provider: Optional[str] = None
    shorthand: Optional[str] = None
    display: Optional[str] = None
    prefix: Optional[str] = None
    logo: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    default_confidence: float = 0.5


class ModelRegistry:
    """Registry for OCR model configurations."""

    def __init__(self, config_path: str = "configs/batch_pipeline.yaml"):
        """Initialize the model registry from configuration YAML."""
        self.models = {}
        self.default_model = None
        self.load_from_config(config_path)

    def load_from_config(self, config_path: str) -> None:
        """Load model registry from configuration file."""
        config = load_config(config_path)

        # Process models from the registry
        if "models_registry" in config:
            for model_entry in config["models_registry"]:
                model_config = ModelConfig(**model_entry)
                self._infer_missing_properties(model_config)

                # Add to registry by name and shorthand
                self.models[model_config.name] = model_config
                if model_config.shorthand:
                    self.models[model_config.shorthand] = model_config

        # Process selected models list and set default
        if "ocr" in config and "selected_models" in config["ocr"]:
            selected = config["ocr"]["selected_models"]
            if selected and selected[0] in self.models:
                self.default_model = self.models[selected[0]]

        # Fallback for default model
        if not self.default_model and self.models:
            self.default_model = next(iter(self.models.values()))

    def _infer_missing_properties(self, model_config: ModelConfig) -> None:
        """Fill in missing properties based on model name patterns."""
        if not model_config.display:
            model_config.display = self._generate_display_name(model_config.name)

        if not model_config.prefix:
            model_config.prefix = self._generate_prefix(model_config.display)

        if not model_config.logo:
            model_config.logo = self._infer_logo(model_config.name)

    def _infer_logo(self, model_name: str) -> str:
        """Infer the logo based on the model name."""
        model_name = model_name.lower()

        if any(n in model_name for n in ["gpt", "openai"]):
            return "openai.svg"
        elif any(n in model_name for n in ["mistral", "pixtral"]):
            return "mistral.svg"
        elif "gemma" in model_name:
            return "gemma.svg"
        elif "llava" in model_name:
            return "microsoft.svg"
        elif any(n in model_name for n in ["moondream", "phi", "granite"]):
            return "ollama.svg"

        return "default.svg"

    def _generate_display_name(self, model_name: str) -> str:
        """Generate a human-readable display name."""
        if "/" in model_name:
            model_name = model_name.split("/")[1]

        parts = re.split(r"[-_:.]", model_name)

        formatted = []
        for part in parts:
            if re.match(r"^\d+b$", part.lower()):  # Size (7b, 11b)
                formatted.append(part.upper())
            elif re.match(r"^\d+(\.\d+)*$", part):  # Version numbers
                formatted.append(part)
            elif part.lower() in ["gpt", "llm"]:
                formatted.append(part.upper())
            else:
                formatted.append(part.capitalize())

        return " ".join(formatted)

    def _generate_prefix(self, display_name: str) -> str:
        """Generate a file prefix from display name."""
        prefix = display_name.lower().replace(" ", "_").replace("-", "_")
        prefix = re.sub(r"[^a-z0-9_]", "", prefix)
        return prefix

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID or shorthand."""
        return self.models.get(model_id)

    def get_model_by_prefix(self, prefix: str) -> Optional[ModelConfig]:
        """Get a model configuration by its prefix."""
        for model in self.models.values():
            if model.prefix == prefix:
                return model
        return None


def get_model_info(model_name: str) -> Tuple[str, str]:
    """Returns a tuple (display, prefix) for a given model name.

    Args:
        model_name: The name of the model

    Returns:
        A tuple (display, prefix)
    """
    model = model_registry.get_model(model_name)
    if model:
        return model.display, model.prefix

    # Generate fallback values
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


def get_model_prefix(model_name: str) -> str:
    """Get standardized prefix from model name."""
    if "/" in model_name:
        model_name = model_name.split("/")[1]

    if ":" in model_name:
        model_name = model_name.replace(":", "_")

    prefix = model_name.lower().replace("-", "_").replace(".", "_")
    prefix = re.sub(r"[^a-z0-9_]", "", prefix)
    return prefix


# global instance of the ModelRegistry
model_registry = ModelRegistry()

# Export the registry's models dict for compatibility
MODEL_CONFIGS = model_registry.models

# Export the default model for compatibility
DEFAULT_MODEL = model_registry.default_model
