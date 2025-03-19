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
Utilities for loading and applying Docker settings to ZenML steps.
"""

from typing import Any, Dict, Optional

from zenml.config import DockerSettings

from .config_loaders import load_config
from .logger import logger


def load_docker_settings_for_step(
    step_name: str,
    config_path: Optional[str] = "configs/remote_finetune.yaml",
) -> Optional[DockerSettings]:
    """
    Load Docker settings for a specific step from the config file.

    Args:
        step_name: Name of the step to load Docker settings for
        config_path: Optional path to config file (will use default if None)

    Returns:
        DockerSettings object or None if no settings found for the step
    """
    config_data = load_config(config_path)

    global_docker_settings = config_data.get("settings", {}).get("docker", {})

    step_settings = config_data.get("steps", {}).get(step_name, {})
    step_docker_settings = step_settings.get("settings", {}).get("docker", {})

    if not step_docker_settings and not global_docker_settings:
        logger.log_warning(f"No Docker settings found for step '{step_name}'")
        return None

    merged_settings = {**global_docker_settings, **step_docker_settings}

    return DockerSettings(**merged_settings)


def apply_docker_settings(
    step_name: str,
    config_path: Optional[str] = "configs/remote_finetune.yaml",
) -> Dict[str, Any]:
    """
    Helper function to directly generate the settings dictionary for a ZenML step decorator.

    Args:
        step_name: Name of the step to load Docker settings for
        config_path: Optional path to config file (will use default if None)

    Returns:
        Dictionary that can be directly used in the settings parameter of a step decorator
    """
    docker_settings = load_docker_settings_for_step(step_name, config_path)

    if docker_settings:
        return {"docker": docker_settings}
    return {}
