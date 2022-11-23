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
from pipelines.deployment_pipeline import yolov5_deployment_pipeline
from pipelines.inference_pipeline import yolov5_inference_pipeline
from pipelines.train_pipeline import yolov5_pipeline
from steps import (
    PredictionServiceLoaderStepParameters,
    bento_builder,
    bentoml_model_deployer,
    bentoml_prediction_service_loader,
    data_loader,
    deployment_trigger,
    inference_loader,
    model_loader,
    predictor,
    train_augmenter,
    trainer,
    valid_augmenter,
)

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
        training_pipeline = yolov5_pipeline(
            data_loader=data_loader(),
            train_augmenter=train_augmenter(),
            valid_augmenter=valid_augmenter(),
            trainer=trainer(),
        )
        training_pipeline.run()
    if deploy:
        deployment_pipeline = yolov5_deployment_pipeline(
            model_loader=model_loader(),
            deployment_trigger=deployment_trigger(),
            bento_builder=bento_builder,
            deployer=bentoml_model_deployer,
        )
        deployment_pipeline.run()
    if predict:
        inference_pipeline = yolov5_inference_pipeline(
            inference_loader=inference_loader(),
            prediction_service_loader=bentoml_prediction_service_loader(
                params=PredictionServiceLoaderStepParameters(
                    model_name="sign_language_yolov5",
                    pipeline_name="yolov5_deployment_pipeline",
                    step_name="bentoml_model_deployer_step",
                )
            ),
            predictor=predictor(),
        )
        inference_pipeline.run()


if __name__ == "__main__":
    main()
