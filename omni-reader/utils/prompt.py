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
"""This module contains the prompt and schema for the OCR model."""

from typing import Optional

from pydantic import BaseModel, Field


class ImageDescription(BaseModel):
    """Base model for OCR results."""

    raw_text: str
    confidence: Optional[float] = None


def get_prompt(custom_prompt: Optional[str] = None) -> str:
    """Default prompt for the OCR model."""
    if custom_prompt:
        return custom_prompt
    return """Extract all visible text from this image **without any changes**.
        - **Do not summarize, paraphrase, or infer missing text.**
        - Retain all spacing, punctuation, and formatting exactly as in the image.
        - If text is unclear or partially visible, extract as much as possible without guessing.
        - **Include all text, even if it seems irrelevant or repeated.** 
        - Return your response as a JSON object with the following fields:
            - raw_text: The extracted text from the image
            - confidence: The confidence score in the extracted text as a float between 0 and 1
        """


def get_json_prompt() -> str:
    """Alternative prompt focusing on JSON output."""
    return """Extract all text from this image and format it as JSON, **strictly preserving** the structure.
        - **Do not summarize, add, or modify any text.**
        - Maintain hierarchical sections and subsections as they appear.
        - Include all text, even if fragmented, blurry, or unclear.
        - Use the following keys:
            - raw_text: The extracted text from the image
            - confidence: The confidence score in the extracted text as a float between 0 and 1
        """
