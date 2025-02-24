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
import click
from pipelines.deployment_pipeline import (
    sign_language_detection_deployment_pipeline,
)
from pipelines.inference_pipeline import (
    sign_language_detection_inference_pipeline,
)
from pipelines.train_pipeline import sign_language_detection_train_pipeline

TRAIN = "train"
DEPLOY = "deploy"
PREDICT = "predict"
TRAIN_AND_DEPLOY_AND_PREDICT = "train_and_deploy_and_predict"


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([TRAIN, DEPLOY, PREDICT, TRAIN_AND_DEPLOY_AND_PREDICT]),
    default="None",
    help="Optionally you can choose to only run the deployment "
    "pipeline to train and deploy a model (`train`), or to "
    "only run a prediction against the deployed model "
    "(`deploy`). By default both will be run "
    "(`train_and_deploy`).",
)
def main(
    config: str,
):
    train = config == TRAIN or config == TRAIN_AND_DEPLOY_AND_PREDICT
    deploy = config == DEPLOY or config == TRAIN_AND_DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == TRAIN_AND_DEPLOY_AND_PREDICT

    if train:
        training_pipeline = sign_language_detection_train_pipeline()
        training_pipeline.run()
    if deploy:
        deployment_pipeline = sign_language_detection_deployment_pipeline()
        deployment_pipeline.run()
    if predict:
        inference_pipeline = sign_language_detection_inference_pipeline()
        inference_pipeline.run()


if __name__ == "__main__":
    main()
