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
from typing import List, Optional, Any

import click
from zenml.client import Client
from zenml.config.schedule import Schedule

from zennews.models import Article


def error(text: str) -> None:
    """Wrapper around the click.ClickException.

    Args:
        text, str, the exception text.

    Raises:
        click.ClickException with the defined style.
    """
    raise click.ClickException(message=click.style(text, fg="red", bold=True))


def warning(text: str) -> None:
    """Wrapper around the 'click.echo' for warning messages.

    Args:
        text, str, the warning message.
    """
    click.secho(text, fg='yellow', bold=True)


def title(text: str) -> None:
    click.secho(f"\n ----- {text.upper()} ----- \n", fg='cyan', bold=True)


class stack_handler(object):
    """Context manager that switches the active stack temporarily."""

    def __init__(self, target_stack_name: str = 'default') -> None:
        """Initialization of the stack handler.

        Args:
            target_stack_name: str, the name of the target stack
        """
        self.active_stack_name = None
        self.target_stack_name = target_stack_name

    def __enter__(self) -> "stack_handler":
        """Enter function of the stack handler.

        Saves the name of the current active stack and activates the temporary
        target stack.

        Returns:
            the handler instance.
        """
        client = Client()

        self.active_stack_name = client.active_stack_model.name

        if self.active_stack_name != self.target_stack_name:
            warning(
                "Temporarily changing the active stack from "
                f"{self.active_stack_name} to {self.target_stack_name}!"
            )

        client.activate_stack(self.target_stack_name)
        return self

    def __exit__(
        self,
        type_: Optional[Any],
        value: Optional[Any],
        traceback: Optional[Any],
    ) -> Any:
        """Exit function of the stack handler.

        Sets the previous stack as the active stack again.

        Args:
            type_: The class of the exception
            value: The instance of the exception
            traceback: The traceback of the exception
        """

        client = Client()

        if self.active_stack_name != self.target_stack_name:
            warning(
                "Changing the active stack back from "
                f"{self.active_stack_name} to {self.target_stack_name}!"
            )

        client.activate_stack(self.active_stack_name)


def build_pipeline(source_step, source_params, **kwargs):
    """Create an instance of a zennews pipeline with the given steps.

    Args:
        source_step: the step class for the source
        source_params: the parameters class for the source
        kwargs: the parameters

    Returns:
        an instance of a zennews pipeline
    """

    from zennews.pipelines import zen_news_pipeline
    from zennews.steps import bart_large_cnn_samsum, post_summaries

    pipeline = zen_news_pipeline(
        collect=source_step(source_params.parse_obj(kwargs)),
        summarize=bart_large_cnn_samsum(),
        report=post_summaries(),
    )

    return pipeline


def parse_schedule(frequency: str, flavor: str) -> Schedule:
    """Create a schedule object that can be used by the orchestrator flavor.

    Args:
        frequency: str, the frequency set for that profile
        flavor: str, the flavor of the orchestrator

    Returns:
         the proper Schedule object
    """
    if flavor == "vertex":
        if frequency == 'debug':
            return Schedule(cron_expression="*/5 * * * *")
        elif frequency == 'hourly':
            return Schedule(cron_expression="0 * * * *")
        elif frequency == 'daily':
            return Schedule(cron_expression="0 9 * * *")
        elif frequency == 'weekly':
            return Schedule(cron_expression="0 9 * * MON")
        else:
            raise ValueError('Please use one of the supported values.')

    else:
        raise NotImplementedError(
            'The schedule parser can only be used by the Vertex orchestrator!'
        )


def display_summaries(summaries: str) -> None:
    """Display the articles on the CLI."""
    from rich.markdown import Markdown
    from rich.console import Console

    md = Markdown(summaries)
    Console().print(md)
