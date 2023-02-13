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


from pydantic import BaseModel


class Article(BaseModel):
    """Base model for articles with full text.

    Attributes:
        source: the name of the source,, e.g. BBC.
        section: a tag to give the article, e.g. a category.
        url: the url of the original article.
        text: the text which represents the article.
    """
    source: str
    section: str
    url: str
    text: str
