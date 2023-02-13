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

from typing import Type

from zenml.alerter.base_alerter import (
    BaseAlerterConfig,
    BaseAlerterFlavor,
    BaseAlerter,
)


class DiscordAlerterConfig(BaseAlerterConfig):
    """Configuration for the discord webhook alerter.

    Attributes:
        webhook_url_secret: str, the name of the secret which holds the
            url of the webhook.
    """
    webhook_url_secret: str


class DiscordAlerterFlavor(BaseAlerterFlavor):
    """Discord Alerter flavor."""

    @property
    def name(self):
        """Name of the discord flavor.

        Returns:
            the name of the discord flavor.
        """
        return "discord-webhook"

    @property
    def config_class(self) -> Type[BaseAlerterConfig]:
        """The configuration class for the discord alerter.

        Returns:
            The config class.
        """
        return DiscordAlerterConfig

    @property
    def implementation_class(self) -> Type[BaseAlerter]:
        """The implementation class for the discord alerter.

        Returns:
            The alerter class.
        """
        from zennews.alerter.discord_alerter import DiscordAlerter

        return DiscordAlerter
