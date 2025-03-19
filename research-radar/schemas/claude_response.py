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

from typing import Optional

from pydantic import BaseModel, Field


class ClaudeResponse(BaseModel):
    """
    Schema for Claude response
    """

    prediction: int
    latency: float
    input_tokens: int
    output_tokens: int
    raw_response: str
    confidence: float = Field(default=-1)
    reason: str = Field(default="")
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        return self.input_tokens * 0.00025 + self.output_tokens * 0.00075
