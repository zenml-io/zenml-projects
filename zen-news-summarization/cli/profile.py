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

from typing import List

import click

from cli.base import cli
from cli.utils import parse_args, save_profile, load_profiles
from models.profile import Profile


@cli.group('profile')
def profile() -> None:
    """Base group for ZenNews profiles."""


@profile.command('create', context_settings={"ignore_unknown_options": True})
@click.argument("name")
@click.argument("source")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def create_profile(
        name: str,
        source: str,
        stack: str,
        frequency: str,
        args: List[str] = None,
) -> None:
    """Create a ZenNews profile.

    Arguments:
        name: str, the name given to the profile.
        source: str, the name of the news source like "bbc" or "cnn".
        stack: str, the name of the stack which will be associated with the
            profile.
        frequency: str, the frequency which will be used to schedule the
            pipeline, default "1d" one-day.
        args: list of additional arguments which will be parsed to parameterize
            the source step.
    """
    # TODO: Make sure that a profile with the same name doesn't exist already
    # Parse the arguments
    parsed_args = parse_args(source=source, args=args)

    profile = Profile(
        name=name,
        source=source,
        stack=stack,
        schedule=frequency,
        args=parsed_args,
    )

    save_profile(profile)


@profile.command('describe')
@click.option("--name", "-n", default=None, type=str)
def describe_profile(name: str) -> None:
    click.echo('hello')
    print(name)


@profile.command('active')
@click.argument("name")
@click.option("--stack", "-s", default=None, type=str)
@click.option("--frequency", "-f", default='1d', type=str)
def activate_profile() -> None:
    # TODO: Check whether the given source argument is currently supported
    # TODO: Check the validity of the frequency, parse it and create a schedule
    if stack is None:
        from zenml.client import Client
        stack = Client().active_stack_model.name
        # TODO: Check whether the stack has an orchestrator which supports
        #   scheduling and an alerter

    click.echo('hello')


@profile.command("deactivate")
def deactivate_profile() -> None:
    click.echo()


@profile.command('delete')
@click.argument("name")
def delete_profile() -> None:
    click.echo('hello')


@profile.command('list')
def list_profiles() -> None:
    click.echo('hello')


def schedule_pipeline(params):
    from pipelines import zen_news_pipeline

    from steps.sources.bbc import bbc_news_source
    from steps.summarize.bart_large_cnn_samsum import \
        bart_large_cnn_samsum_parameters
    from steps.report.report import post_summaries

    pipeline = zen_news_pipeline(
        collect=bbc_news_source(params),
        summarize=bart_large_cnn_samsum_parameters(),
        report=post_summaries(),
    )
    pipeline.run()
