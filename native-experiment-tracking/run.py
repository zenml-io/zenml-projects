# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import click
from sklearn.utils._param_validation import InvalidParameterError
from zenml import Model
from zenml.client import Client
from zenml.logger import get_logger

from pipelines import training

logger = get_logger(__name__)


@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(
        no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        no_cache: If `True` cache will be disabled.
    """
    client = Client()
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        "training.yaml"
    )
    enable_cache = not no_cache

    alpha_values = [0.0001, 0.001, 0.01]
    penalties = ["l2", "l1", "elasticnet"]
    losses = ["hinge", "squared_hinge", "modified_huber"]
    for penalty in penalties:
        for loss in losses:
            for alpha_value in alpha_values:
                logger.info(f"Training with alpha: {alpha_value}, penalty: {penalty}, loss: {loss}")

                model = Model(
                    name="breast_cancer_classifier",
                    tags=[f"alpha: {alpha_value}", f"penalty: {penalty}", f"loss: {loss}"]
                )
                try:
                    training.with_options(config_path=config_path, enable_cache=enable_cache, model=model)(
                        alpha_value=alpha_value, penalty=penalty, loss=loss)
                except RuntimeError:
                    pass
                else:
                    logger.info("Training pipeline finished successfully!\n\n")


if __name__ == "__main__":
    main()
