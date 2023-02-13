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

from zennews.cli.constants import APP_NAME


@click.group(APP_NAME)
def cli():
    """CLI base command for ZenML."""

    # Supress warning messages during client initializations
    os.environ["ZENML_ENABLE_REPO_INIT_WARNINGS"] = "false"

    click.secho(
        r"""
                 ______          _   _                   
                |___  /         | \ | |                  
                   / / ___ _ __ |  \| | _____      _____ 
                  / / / _ \ '_ \| . ` |/ _ \ \ /\ / / __|
                 / /_|  __/ | | | |\  |  __/\ V  V /\__ \
                /_____\___|_| |_|_| \_|\___| \_/\_/ |___/
                                                                               
                      This is where you get the news.
         """,
        fg="magenta",
    )
