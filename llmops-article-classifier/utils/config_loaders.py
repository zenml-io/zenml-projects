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

"""
Utilities for loading and validating configuration.
"""

import os
from typing import Dict

import dotenv
import yaml
from pydantic import ValidationError

from schemas import validate_config
from utils import logger

dotenv.load_dotenv()


def load_config(config_path: str = "configs/base_config.yaml") -> Dict:
    """
    Load config from YAML file.
    Handles YAML inheritance via the '_extends' property.

    Args:
        config_path: Path to config YAML file

    Returns:
        Dictionary of settings with all inheritance resolved
    """
    if not config_path.startswith("configs/"):
        config_path = os.path.join("configs", config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        validated_config = validate_config(config)
        return validated_config.model_dump()
    except ValidationError as e:
        logger.error(f"Invalid config: {e}")
        raise ValueError(f"Config validation failed: {e}") from e
