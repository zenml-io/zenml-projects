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
"""This module contains the prompt for the OCR model."""

import re
from typing import Optional


def get_prompt(
    custom_prompt: Optional[str] = None,
) -> str:
    """Get the prompt for the OCR model."""
    if custom_prompt:
        return custom_prompt
    return (
        "First, describe the image in detail. "
        "Next, list any physical entities visible in the image (e.g., 'street sign', 'building', 'car', 'tree') "
        "Then, extract all text visible in the image as raw text. "
        "Finally, rate your confidence in the extracted text as a float between 0 and 1."
    )
