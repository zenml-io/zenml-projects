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

import openai
from zenml.client import Client
from zenml.post_execution import PipelineRunView, get_pipeline
from zenml.steps import BaseParameters, StepContext, step


class SummarizerParams(BaseParameters):
    """Prompts for the summarizer."""

    system_content: str = "Act like a data analytics expert."
    prompt_preamble: str = "Summarize the latest data from the database. The data is YouTube video titles and we are keen to understand what sort of videos users are accessing, the trends, and similarities between them. We also have the analysis from last time that we will give to you as input."
    prompt_example: str = "The latest data indicates the following key insights: 1. The number of users has increased by 10% in the last 24 hours. 2. The number of users has increased by 10% in the last 24 hours. 3... Compared to our last summary, it seems our users are starting to watch more videos about cats."


@step(enable_cache=True)
def gpt_4_summarizer(
    params: SummarizerParams, documents: List[str], context: StepContext
) -> str:
    """Summarizes the data using GPT-4."""
    openai_secret = Client().get_secret("openai")

    p: PipelineRunView = get_pipeline(context.pipeline_name)

    try:
        last_step = p.runs[1].get_step("generate_summary")
        if len(last_step.outputs) == 0:
            last_analysis = "No previous analysis found."
        else:
            last_analysis = last_step.output.read()
    except KeyError:
        last_analysis = "No previous analysis found."

    openai.api_key = openai_secret.secret_values["api_key"]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": params.system_content},
            {
                "role": "user",
                "content": params.prompt_preamble
                + "\n"
                + "Here is an example output:"
                + "\n"
                + params.prompt_example
                + "\n"
                + "Here is the data: "
                + str(documents)
                + "\n"
                + "And here is the previous analysis: "
                + last_analysis,
            },
        ],
        temperature=0,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content
