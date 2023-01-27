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
from zenml.cli.utils import parse_name_and_extra_arguments
from typing import List
from cli.base import cli
from models.profile import Profile

from cli.constants import SUPPORTED_SOURCES

def parse_args(source: str, args: List[str]):
    results = {}
    c = None
    for a in args:
        if c is None:
            if a.startswith("--"):
                c = a
                continue
            else:
                raise ValueError(f'nononono {c}{a}')

        if c is not None:
            if a.startswith("--"):
                results[c] = True
                c = a
            else:
                results[c] = a
                c = None

    if c is not None:
        results[c] = True

    return SUPPORTED_SOURCES.get(source)(**results)


@cli.group('profile')
def profile() -> None:
    """"""


@profile.command('create', context_settings={"ignore_unknown_options": True})
@click.argument("name")
@click.option("--source", "-so", default=None, type=str)
@click.option("--stack", "-st", default=None, type=str)
@click.option("--frequency", "-f", default='1d', type=str)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def create_profile(name, source, stack, frequency, args) -> None:
    print(args)
    parsed_args = parse_args(source=source, args=args)
    print(type(parsed_args))

    Profile(
        name=name,
        source=source,
        stack=stack,
        schedule=frequency,
        args=parsed_args,
    )

    click.echo('hello')

    schedule_pipeline(parsed_args)




@profile.command('describe')
@click.option("--name", "-n", default=None, type=str)
def describe_profile(name: str) -> None:
    click.echo('hello')
    print(name)


@profile.command('active')
@click.argument("name")
def activate_profile() -> None:
    click.echo('hello')


@profile.command('delete')
@click.argument("name")
def delete_profile() -> None:
    click.echo('hello')


@profile.command('list')
def list_profiles() -> None:
    click.echo('hello')



def schedule_pipeline(params):
    from pipelines import zen_news_pipeline

    from steps.sources.bbc import  bbc_news_source
    from steps.summarize.bart_large_cnn_samsum import bart_large_cnn_samsum_parameters
    from steps.report.report import post_summaries

    pipeline = zen_news_pipeline(
        collect=bbc_news_source(params),
        summarize=bart_large_cnn_samsum_parameters(),
        report=post_summaries(),
    )
    pipeline.run()