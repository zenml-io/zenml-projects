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

docker_settings = DockerSettings(requirements="requirements.txt")


@pipeline(enable_cache=False)
def community_analysis_pipeline(booming, churned, report) -> None:
    """Defines a pipeline to analyze the community on Orbit

    Args:
        booming: step that manages booming users
        churned: step that manages churned users
        report: step that creates a report from the results
    """
    booming()
    churned()

    report.after(booming)
    report.after(churned)

    report()
