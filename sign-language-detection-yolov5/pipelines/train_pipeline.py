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


from zenml.pipelines import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

docker_settings = DockerSettings(parent_image="ultralytics/yolov5:latest", requirements="./requirements.txt",required_integrations=[MLFLOW])

@pipeline(enable_cache=False, 
    settings={
        "docker": docker_settings,
        "orchestrator.local_docker": {
            "run_args": {
                "device_requests": [{ "device_ids": ["0"], "capabilities": [['gpu']] }],
                "shm_size": 18446744073692774399,
                "ipc_mode": "host",
                "ulimit": [{ "name": "memlock", "soft": -1 },{ "name": "stack", "soft": -1 }],
                }
            }
        }
    )
def yolov5_pipeline(
    data_loader,
    train_augmenter,
    valid_augmenter,
    trainer,
    detector,
):
    train,valid,test = data_loader()
    augmented_trainset = train_augmenter(train)
    augmented_validset = valid_augmenter(valid)
    model = trainer(augmented_trainset,augmented_validset)
    detector = detector(test,model)