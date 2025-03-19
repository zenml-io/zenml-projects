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

from datetime import datetime
from typing import Optional

from pydantic import AnyHttpUrl, BaseModel, Field, field_serializer, field_validator


class ArticleMeta(BaseModel):
    """
    Metadata of InputArticle for article classification
    """

    url: AnyHttpUrl
    title: Optional[str] = Field(default=None)
    published_date: Optional[datetime] = Field(default=None)
    author: Optional[str] = Field(default=None)

    @field_serializer("url")
    def serialize_url(self, url: AnyHttpUrl):
        return str(url)

    @field_serializer("published_date")
    def serialize_date(self, dt: Optional[datetime]):
        return dt.isoformat() if dt else None


class InputArticle(BaseModel):
    """
    Schema for all input articles used for classification and training
    """

    @field_validator("text")
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    text: str
    meta: ArticleMeta
