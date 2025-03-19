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
Utility functions for training setup.
"""

from typing import Dict

import torch

from utils import logger


def determine_device():
    """Determine the appropriate device for training."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = "cpu"
        logger.info("Using CPU for training (no GPU detected)")

    return device


def determine_if_remote(config: Dict):
    """Determine if we're in a remote environment."""
    is_remote = config.get("execution_mode") == "remote"

    if not is_remote and "remote" in config.get("project", {}).get("version", ""):
        is_remote = True
        logger.info("Remote execution mode detected from project version")
    if is_remote:
        logger.info("Remote execution mode enabled")
    else:
        logger.info("Local execution mode enabled")

    return is_remote
