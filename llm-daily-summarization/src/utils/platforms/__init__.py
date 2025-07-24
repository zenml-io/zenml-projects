"""
Platform integrations (fetch + delivery).

Public re-exports:
    DiscordDeliverer, SlackDeliverer, NotionDeliverer
    DiscordClient,   SlackClient
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

# Fetch clients live in the legacy module but we surface them here
from ..chat_clients import DiscordClient, SlackClient  # noqa: E402

# Local deliverers (defined in sibling modules)
from .discord import DiscordDeliverer
from .local import LocalDeliverer
from .notion import NotionDeliverer
from .slack import SlackDeliverer

__all__ = [
    "DiscordDeliverer",
    "SlackDeliverer",
    "NotionDeliverer",
    "LocalDeliverer",
    "DiscordClient",
    "SlackClient",
]


# Optional: lazy-import future platform modules to avoid circulars
def __getattr__(name: str):
    if name in __all__:
        return globals()[name]
    raise AttributeError(name)
