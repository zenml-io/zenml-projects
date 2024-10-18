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
import concurrent
import multiprocessing
import os
import traceback
from itertools import product

import click
from sklearn.utils._param_validation import InvalidParameterError
from zenml import Model
from zenml.client import Client
from zenml.logger import get_logger

from pipelines import training, feature_engineering

logger = get_logger(__name__)


@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--parallel",
    is_flag=True,
    default=False,
    help="Run training across the complete parameter grid in parallel.",
)
@click.option(
    "--single_run",
    is_flag=True,
    default=False,
    help="Run only one permutation of parameters.",
)
def main(
    no_cache: bool = False,
    parallel: bool = False,
    single_run: bool = False
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        no_cache: If `True` cache will be disabled.
        parallel: If `True` multiprocessing will be used for running hyperparameter tuning in parallel
        single_run: if `True` only one training run will be started
    """
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        "training.yaml"
    )
    enable_cache = not no_cache

    # Run the feature engineering pipeline, this way all invocations within the training pipelines
    # will use the cached output from this pipeline
    # feature_engineering()

    # Here is our set of parameters that we want to explore to find the best combination
    alpha_values = [0.0001, 0.001, 0.01]
    penalties = ["l2", "l1", "elasticnet"]
    losses = ["hinge", "squared_hinge", "modified_huber"]


    if single_run:
        train_model(alpha_values[0], penalties[0], losses[0], config_path, enable_cache)
    else:
        # Lets loop over these
        # Create a list of all parameter combinations
        parameter_combinations = list(product(alpha_values, penalties, losses))

        if parallel:
            parallel_training(config_path, enable_cache, parameter_combinations)
        else:
            for alpha_value, penalty, loss in parameter_combinations:
                train_model(alpha_value, penalty, loss, config_path, enable_cache)


def parallel_training(config_path, enable_cache, parameter_combinations):
    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count()
    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(train_model, alpha, penalty, loss, config_path, enable_cache)
                   for alpha, penalty, loss in parameter_combinations]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def train_model(alpha_value: float, penalty: str, loss: str, config_path: str, enable_cache: bool):
    logger.info(f"Training with alpha: {alpha_value}, penalty: {penalty}, loss: {loss}")

    model = Model(
        name="breast_cancer_classifier",
        tags=[f"alpha: {alpha_value}", f"penalty: {penalty}", f"loss: {loss}"]
    )
    try:
        logger.info(f"Starting training with alpha: {alpha_value}, penalty: {penalty}, loss: {loss}")
        training.with_options(
            config_path=config_path, enable_cache=enable_cache, model=model
        )(
            alpha_value=alpha_value, penalty=penalty, loss=loss
        )

        logger.info(f"Training finished successfully for alpha: {alpha_value}, penalty: {penalty}, loss: {loss}")
    except InvalidParameterError:
        logger.info("Pipeline run aborted due to parameter mismatch!\n\n")
        pass
    except Exception as e:
        logger.error(f"Error in training with alpha: {alpha_value}, penalty: {penalty}, loss: {loss}")
        logger.error(f"Exception: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        logger.info("Training pipeline finished successfully!\n\n")


if __name__ == "__main__":
    main()
