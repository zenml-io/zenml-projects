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

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import click
from zenml.client import Client

from pipelines import llm_basic_rag


@click.command(
    help="""
ZenML LLM Complete - Rag Pipeline
"""
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable cache.",
)

@click.option(
    "--create-template",
    "create_template",
    is_flag=True,
    default=False,
    help="Create a run template.",
)
@click.option(
    "--config",
    "config",
    default="rag_local_dev.yaml",
    help="Specify a configuration file"
)
@click.option(
    "--action-id",
    "action_id",
    default=None,
    help="Specify an action ID"
)
def main(
    no_cache: bool = False,
    config: Optional[str]= "rag_local_dev.yaml",
    create_template: bool = False,
    action_id: Optional[str] = None
):
    """
    Executes the pipeline to train a basic RAG model.

    Args:
        no_cache (bool): If `True`, cache will be disabled.
        config (str): The path to the configuration file.
        create_template (bool): If `True`, a run template will be created.
        action_id (str): The action ID.
    """
    client = Client()
    config_path = Path(__file__).parent.parent / "configs" / config

    if create_template:
        # run pipeline
        run = llm_basic_rag.with_options(
            config_path=str(config_path),
            enable_cache=not no_cache
        )()
        # create new run template
        rt = client.create_run_template(
            name=f"production-llm-complete-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            deployment_id=run.deployment_id
        )
        # update the action with the new template
        client.update_action(
            name_id_or_prefix=UUID(action_id),
            configuration={
                "template_id": str(rt.id)
            }
        )

    else:
        llm_basic_rag.with_options(
            config_path=str(config_path),
            enable_cache=not no_cache
        )()


if __name__ == "__main__":
    main()