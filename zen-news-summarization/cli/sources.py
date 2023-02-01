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

import click
from zenml.config.schedule import Schedule
from zenml.enums import StackComponentType

from cli import SUPPORTED_ORCHESTRATORS
from steps import SOURCE_STEP_MAPPING
from cli.base import cli
from typing import Dict, Any
from cli.utils import warning, error, stack_handler, display_articles
from datetime import datetime

from zenml.client import Client


def generate_single_source_command(
    source_name: str, source_dict: Dict[Any, Any]
):
    """Function to generate dynamic Click commands for the ZenNews CLI."""

    source_step = source_dict['step']
    source_params = source_dict['parameters']

    @click.command(
        name=source_name,
        help=f'Run/Schedule ZenNews pipelines for articles '
             f'from {source_name.upper()}.'
    )
    @click.option(
        '--schedule',
        type=int,
        default=None,
        help="The amount of minutes to set as the frequency while scheduling."
    )
    @click.option(
        '--stack',
        type=str,
        default=None,
        help="The stack to use when scheduling the pipeline."
    )
    @click.option(
        '--force',
        '-f',
        is_flag=True,
        default=False,
        help="Flag to skip the confirmation."
    )
    def source_command(
        schedule: str,
        stack: str,
        force: bool,
        **kwargs: Dict[Any, Any],
    ) -> None:
        """Runs a ZenNews pipeline to summarize news articles.

        Args:
            schedule: str, the frequency to use if/when scheduling this
                pipeline.
            stack: str, the name of the stack to schedule the pipeline with.
            force: bool, flag to skip the confirmation if there is no
                schedule.
        """
        if schedule:
            # If there is a schedule, figure out which stack to use
            client = Client()
            stack_name = stack or client.active_stack_model.name

            with stack_handler(stack_name):
                active_stack = client.active_stack_model
                active_components = active_stack.components
                orchestrator = active_components[StackComponentType.ORCHESTRATOR][0]

                if orchestrator.flavor not in SUPPORTED_ORCHESTRATORS:
                    error('Does not work.')

                if StackComponentType.ALERTER not in active_components:
                    warning('There is no alerter!')

                pipeline = build_pipeline(
                    source_step=source_step,
                    source_params=source_params,
                    **kwargs
                )

                run_name = f'news_{source_name}_{{date}}_{{time}}'

                # TODO: Implement a proper parser
                # schedule_obj = parse_schedule(schedule, orchestrator.flavor)
                schedule_obj = Schedule(cron_expression='*/15 * * * *')

                pipeline.run(run_name=run_name, schedule=schedule_obj)



        else:
            # If there is no schedule, RUN the pipeline locally
            if not force:
                warning(
                    'Testing a profile will temporarily change your active '
                    'stack to the default stack and run the pipeline '
                    'locally. This also means that the pipeline will '
                    'download and utilize the model locally.'
                )
                if not click.confirm('Would you like to continue?'):
                    error('Stopped the process.')

            # Build and run the pipeline
            from pipelines import zen_news_pipeline
            from steps import bart_large_cnn_samsum, post_summaries

            with stack_handler("default"):
                pipeline = zen_news_pipeline(
                    collect=source_step(source_params.parse_obj(kwargs)),
                    summarize=bart_large_cnn_samsum(),
                    report=post_summaries(),
                )
                run_name = f'test_{source_name}' \
                           f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}'

                pipeline.run(run_name=run_name)

            # Get the run view of the pipeline and showcase the results
            from zenml.post_execution import get_run

            run_view = get_run(run_name)
            step_view = run_view.get_step("report")
            artifact_view = step_view.outputs['output']
            summaries = artifact_view.read()
            display_articles(summaries)

    # Extract and add the step parameters to the Click command
    properties = source_params.schema()['properties']
    for property_name, property_values in properties.items():

        property_type = property_values.get('type', None)
        property_default = property_values.get('default', None)
        property_description = property_values.get('description', '')

        if property_type == 'boolean':
            option = click.option(
                f"--{property_name}",
                is_flag=True,
                default=property_default,
                help=property_description,
            )
        else:
            option = click.option(
                f"--{property_name}",
                type=str,
                default=property_default,
                help=property_description,
            )

        # Add it
        option(source_command)

    cli.add_command(source_command)


def generate_all_source_commands():
    # Iterate over all the sources available/implemented
    for s_name, s_dict in SOURCE_STEP_MAPPING.items():
        # Create all the subcommands
        generate_single_source_command(s_name, s_dict)


generate_all_source_commands()


def build_pipeline(source_step, source_params, **kwargs):
    from pipelines import zen_news_pipeline
    from steps import bart_large_cnn_samsum, post_summaries

    pipeline = zen_news_pipeline(
        collect=source_step(source_params.parse_obj(kwargs)),
        summarize=bart_large_cnn_samsum(),
        report=post_summaries(),
    )

    return pipeline
