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
from pipelines import inference_pipeline, training_pipeline


@click.command()
@click.option(
    "--train",
    "pipeline",
    flag_value="train",
    default=True,
    help="Run the training pipeline.",
)
@click.option(
    "--inference",
    "pipeline",
    flag_value="inference",
    help="Run the inference pipeline.",
)
@click.option(
    "--rerun",
    is_flag=True,
    default=False,
)
def main(pipeline, rerun):
    """Simple CLI interface for annotation example."""
    if pipeline == "train":
        training_pipeline()
    elif pipeline == "inference":
        inference_pipeline(rerun=rerun)


if __name__ == "__main__":
    main()
