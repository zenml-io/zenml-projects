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
"""Schemas for OCR results."""

from typing import Dict, List

from pydantic import BaseModel, RootModel


class OCRResult(BaseModel):
    """OCR result for a single image."""

    id: int
    image_name: str
    raw_text: str
    processing_time: float
    confidence: float


class OCRResultMapping(RootModel):
    """Each model name maps to a list of OCRResult entries."""

    root: Dict[str, List[OCRResult]]
