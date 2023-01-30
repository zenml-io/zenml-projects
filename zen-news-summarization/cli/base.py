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
import os

import click

from cli.constants import APP_NAME, PROFILES_PATH, CONFIG_PATH, APP_DIR
from models import Config
from cli.utils import save_config


@click.group(APP_NAME, invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """CLI base command for ZenML."""

    if not os.path.exists(PROFILES_PATH):
        os.makedirs(PROFILES_PATH)

    if not os.path.exists(CONFIG_PATH):
        save_config(Config())

    if not ctx.invoked_subcommand:
        click.echo(
            r"""
                     ______          _   _                   
                    |___  /         | \ | |                  
                       / / ___ _ __ |  \| | _____      _____ 
                      / / / _ \ '_ \| . ` |/ _ \ \ /\ / / __|
                     / /_|  __/ | | | |\  |  __/\ V  V /\__ \
                    /_____\___|_| |_|_| \_|\___| \_/\_/ |___/
                                                                                   
                          This is where you get the news.
             """
        )
        # TODO: If there is no invoked subcommands, figure out the latest
        #   pipeline and utilize its final artifact to display summaries


@click.command("test")
def test():
    """CLI command to test the news summarization on the default stack."""
    # TODO: Run the pipeline on the default stack for showcasing purposes
    # TODO: Add confirmation


@click.command("clean")
def clean():
    """CLI command to clean the profiles folders and the config file."""
    # TODO: Implement a small clean up method.
    # TODO: Add confirmation
