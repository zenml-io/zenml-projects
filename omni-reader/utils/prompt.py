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

from typing import List, Optional

from pydantic import BaseModel, Field


class ImageDescription(BaseModel):
    """Base model for OCR results."""

    raw_text: str = Field(description="Extracted text from the image")
    confidence: float = Field(default=0.0, description="Confidence score (0-1) for the extraction")


def get_prompt(custom_prompt: Optional[str] = None, language: str = "English") -> str:
    """Return the prompt for the OCR model."""
    if custom_prompt:
        return custom_prompt
    return f"""Extract all visible text from this image in {language} **without any changes**.
        - **Do not summarize, paraphrase, or infer missing text.**
        - Retain all spacing, punctuation, and formatting exactly as in the image.
        - If text is unclear or partially visible, extract as much as possible without guessing.
        - **Include all text, even if it seems irrelevant or repeated.** 
        - Then, rate your confidence in the extracted text as a float between 0 and 1.
        """
