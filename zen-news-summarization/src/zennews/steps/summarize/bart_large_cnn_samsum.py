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

from zenml.steps import step

from zennews.models import Article


@step
def bart_large_cnn_samsum(articles: List[Article]) -> List[Article]:
    """Step that generates summaries of the given list of news articles."""
    from transformers import AutoTokenizer, BartForConditionalGeneration

    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-cnn"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-cnn"
    )

    summarizations = []
    for a in articles:
        inputs = tokenizer(
            [a.text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )

        # Generate Summary
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=2,
            min_length=0,
            max_length=256,
        )

        summary = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        summarizations.append(
            Article(
                source=a.source,
                section=a.section,
                url=a.url,
                text=summary,
            )
        )

    return summarizations
