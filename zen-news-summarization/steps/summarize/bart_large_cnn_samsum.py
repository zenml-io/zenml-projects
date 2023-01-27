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

from typing import List

from transformers import pipeline
from zenml.steps import step

from models import Article, Summary


@step
def bart_large_cnn_samsum_parameters(articles: List[Article]) -> List[Summary]:
    """Step that generates summaries of the given list of news articles."""
    # Create the torch summarizer pipeline
    summarizer = pipeline(
        task="summarization",
        model="philschmid/bart-large-cnn-samsum"
    )

    # Use it for each given article to create summaries
    summarizations = []
    for a in articles:
        summarizations.append(Summary(
            text=summarizer(a),
            url=a.url,
            section=a.section,
        ))

    return summarizations
