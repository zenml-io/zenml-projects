#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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

import json
import os
from typing import Type

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from zennews.models import Article


class ArticleMaterializer(BaseMaterializer):
    """Custom materializer implementation for articles."""
    ASSOCIATED_TYPES = (Article,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[Article]) -> Article:
        """Read an article from the artifact store"""
        super().load(data_type)
        with fileio.open(os.path.join(self.uri, 'article.json'), 'r') as f:
            return Article.parse_raw(json.load(f))

    def save(self, my_obj: Article) -> None:
        """Write an article to artifact store"""
        super().save(my_obj)
        with fileio.open(os.path.join(self.uri, 'article.json'), 'w') as f:
            json.dump(my_obj.json(), f)
