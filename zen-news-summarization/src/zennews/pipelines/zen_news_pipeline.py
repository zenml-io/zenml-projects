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
from zenml.config import DockerSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(requirements=["zennews"], copy_files=False)


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def zen_news_pipeline(collect, summarize, report) -> None:
    """Defines an inference pipeline that summarizes a set of collected news.

    Args:
        collect: step which collects news from a specified source.
        summarize: step that uses an ML model to summarize the gathered news.
        report: step responsible for generating and sending a report from the
            provided summaries.
    """
    news = collect()
    summaries = summarize(news)
    report(summaries)
