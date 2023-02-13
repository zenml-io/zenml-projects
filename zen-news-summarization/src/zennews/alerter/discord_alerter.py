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

from typing import Optional, cast, Any

from discord import SyncWebhook
from zenml.alerter import BaseAlerter
from zenml.client import Client
from zenml.steps import BaseParameters

from zennews.alerter.discord_alerter_flavor import DiscordAlerterConfig


class DiscordAlerter(BaseAlerter):
    """Send messages to a Discord channel."""

    @property
    def config(self) -> DiscordAlerterConfig:
        """Get the config of this artifact store.
        Returns:
            The config of this artifact store.
        """
        return cast(DiscordAlerterConfig, self._config)

    def _get_webhook(self) -> SyncWebhook:
        """Create and return the discord webhook.

        Returns:
            the discord webhook object.
        """

        secret_manager = Client().active_stack.secrets_manager

        webhook_url = secret_manager.get_secret(
            self.config.webhook_url_secret
        ).content["WEBHOOK_URL"]

        return SyncWebhook.from_url(webhook_url)

    def post(
        self, message: Any, params: Optional[BaseParameters]
    ) -> bool:
        """Post a message to a Discord channel using a webhook.

        Args:
            message: str, message to be posted.
            params: parameters passed to the step (unused - will be deprecated).

        Returns:
            True if operation succeeded, else False.
        """

        webhook = self._get_webhook()

        # For the context of this project, the following implementation of the
        # discord alerter is very custom tailored to ZenNews. In order to
        # utilize this alerter in other projects, you would have to generalize
        # and change the implementation.

        from datetime import datetime
        from discord import Embed
        content = f'**From {message[0].source.upper()} generated ' \
                  f'at {datetime.now().strftime("%m/%d/%Y %H:%M:%S")}**'

        embeds = []
        for article in message:
            embeds.append(
                Embed(
                    description=article.text,
                    color=10181046,
                    url=article.url,
                    title=f"**{article.section}**",
                )
            )
        webhook.send(content=content, embeds=embeds)

        return True

    def ask(
        self, message: Any, params: Optional[BaseParameters]
    ) -> bool:
        """Post a message to a Discord channel and wait for approval.

        Args:
            message: str, initial message to be posted.
            params: parameters passed to the step (unused - will be deprecated).

        Returns:
            True if a user approved the operation, else False.
        """
        raise NotImplementedError(
            "The ask function of this alerter is not yet implemented."
        )
