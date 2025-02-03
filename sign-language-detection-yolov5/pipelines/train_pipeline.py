#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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
from zenml.integrations.constants import GCP, MLFLOW
from zenml.pipelines import pipeline

from steps import data_loader, train_augmenter, valid_augmenter, trainer

docker_settings = DockerSettings(
    parent_image="ultralytics/yolov5:latest",
    required_integrations=[MLFLOW, GCP],
    requirements="./requirements.txt",
    dockerignore=".dockerignore",
)


@pipeline(
    enable_cache=True,
    settings={
        "docker": docker_settings,
    },
)
def sign_language_detection_train_pipeline():
    train, valid, test = data_loader()
    augmented_trainset = train_augmenter(train)
    augmented_validset = valid_augmenter(valid)
    trainer(augmented_trainset, augmented_validset)
