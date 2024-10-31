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
import yaml
from pipelines.llm_basic_rag import llm_basic_rag
from zenml.client import Client
from zenml.exceptions import ZenKeyError


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
    help="Specify a configuration file",
)
@click.option(
    "--service-account-id",
    "service_account_id",
    default=None,
    help="Specify a service account ID",
)
@click.option(
    "--event-source-id",
    "event_source_id",
    default=None,
    help="Specify an event source ID",
)
def main(
    no_cache: bool = False,
    config: Optional[str] = "rag_local_dev.yaml",
    create_template: bool = False,
    service_account_id: Optional[str] = None,
    event_source_id: Optional[str] = None,
):
    """
    Executes the pipeline to train a basic RAG model.

    Args:
        no_cache (bool): If `True`, cache will be disabled.
        config (str): The path to the configuration file.
        create_template (bool): If `True`, a run template will be created.
        action_id (str): The action ID.
        service_account_id (str): The service account ID.
        event_source_id (str): The event source ID.
    """
    client = Client()
    config_path = Path(__file__).parent / "configs" / config

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if create_template:
        # run pipeline
        run = llm_basic_rag.with_options(
            config_path=str(config_path), enable_cache=not no_cache
        )()
        # create new run template
        rt = client.create_run_template(
            name=f"production-llm-complete-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            deployment_id=run.deployment_id,
        )

        try:
            # Check if an action ahs already be configured for this pipeline
            action = client.get_action(
                name_id_or_prefix="LLM Complete (production)",
                allow_name_prefix_match=True,
            )
        except ZenKeyError:
            if not event_source_id:
                raise RuntimeError(
                    "An event source is required for this workflow."
                )

            if not service_account_id:
                service_account_id = client.create_service_account(
                    name="github-action-sa",
                    description="To allow triggered pipelines to run with M2M authentication.",
                ).id

            action_id = client.create_action(
                name="LLM Complete (production)",
                configuration={
                    "template_id": str(rt.id),
                    "run_config": pop_restricted_configs(config),
                },
                service_account_id=service_account_id,
                auth_window=0,
            ).id
            client.create_trigger(
                name="Production Trigger LLM-Complete",
                event_source_id=UUID(event_source_id),
                event_filter={"event_type": "tag_event"},
                action_id=action_id,
                description="Trigger pipeline to reindex everytime the docs are updated through git.",
            )
        else:
            # update the action with the new template
            # here we can assume the trigger is fully set up already
            client.update_action(
                name_id_or_prefix=action.id,
                configuration={
                    "template_id": str(rt.id),
                    "run_config": pop_restricted_configs(config),
                },
            )

    else:
        llm_basic_rag.with_options(
            config_path=str(config_path), enable_cache=not no_cache
        )()


def pop_restricted_configs(run_configuration: dict) -> dict:
    """Removes restricted configuration items from a run configuration dictionary.

    Args:
        run_configuration: Dictionary containing run configuration settings

    Returns:
        Modified dictionary with restricted items removed
    """
    # Pop top-level restricted items
    run_configuration.pop("parameters", None)
    run_configuration.pop("build", None)
    run_configuration.pop("schedule", None)

    # Pop docker settings if they exist
    if "settings" in run_configuration:
        run_configuration["settings"].pop("docker", None)

    # Pop docker settings from steps if they exist
    if "steps" in run_configuration:
        for step in run_configuration["steps"].values():
            if "settings" in step:
                step["settings"].pop("docker", None)

    return run_configuration


if __name__ == "__main__":
    main()
