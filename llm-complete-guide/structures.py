#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Document:
    """Custom dataclass to represent a document.

    Attributes:
        page_content: The content of the document.
        filename: The filename or URL (for web docs) of the document.
        parent_section: The parent section of the document.
        url: The URL of the document (if web-derived).
        embedding: The embedding of the document.
        token_count: The number of tokens in the document.
    """

    page_content: str
    filename: Optional[str] = None
    parent_section: Optional[str] = None
    url: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    token_count: Optional[int] = None