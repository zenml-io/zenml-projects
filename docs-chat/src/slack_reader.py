"""Slack reader."""
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

from llama_index import Document
from llama_index.readers.base import BaseReader

logger = logging.getLogger(__name__)


class SlackReader(BaseReader):
    """Slack reader.
    Reads conversations from channels.
    Args:
        slack_token (Optional[str]): Slack token. If not provided, we
            assume the environment variable `SLACK_BOT_TOKEN` is set.
        earliest_date (Optional[datetime]): Earliest date from which
            to read conversations. If not provided, we read all messages.
    """

    def __init__(
        self,
        slack_token: Optional[str] = None,
        earliest_date: Optional[datetime] = None,
    ) -> None:
        """Initialize with parameters."""
        from slack_sdk import WebClient

        if slack_token is None:
            slack_token = os.environ["SLACK_BOT_TOKEN"]
        if slack_token is None:
            raise ValueError(
                "Must specify `slack_token` or set environment "
                "variable `SLACK_BOT_TOKEN`."
            )
        self.client = WebClient(token=slack_token)
        self.earliest_date_timestamp = earliest_date.timestamp()
        res = self.client.api_test()
        if not res["ok"]:
            raise ValueError(f"Error initializing Slack API: {res['error']}")

    def _read_message(self, channel_id: str, message_ts: str) -> str:
        from slack_sdk.errors import SlackApiError

        """Read a message."""

        messages_text = []
        next_cursor = None
        while True:
            try:
                # https://slack.com/api/conversations.replies
                # List all replies to a message, including the message itself.
                if self.earliest_date is None:
                    result = self.client.conversations_replies(
                        channel=channel_id, ts=message_ts, cursor=next_cursor
                    )
                else:
                    result = self.client.conversations_replies(
                        channel=channel_id,
                        ts=message_ts,
                        cursor=next_cursor,
                        oldest=self.earliest_date_timestamp,
                    )
                messages = result["messages"]
                messages_text.extend(message["text"] for message in messages)
                if not result["has_more"]:
                    break

                next_cursor = result["response_metadata"]["next_cursor"]
            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    logger.error(
                        "Rate limit error reached, sleeping for: {} seconds".format(
                            e.response.headers["retry-after"]
                        )
                    )
                    time.sleep(int(e.response.headers["retry-after"]))
                else:
                    logger.error(
                        "Error parsing conversation replies: {}".format(e)
                    )

        return "\n\n".join(messages_text)

    def _read_channel(
        self, channel_id: str, reverse_chronological: bool
    ) -> str:
        from slack_sdk.errors import SlackApiError

        """Read a channel."""

        result_messages = []
        next_cursor = None
        while True:
            try:
                # Call the conversations.history method using the WebClient
                # conversations.history returns the first 100 messages by default
                # These results are paginated,
                # see: https://api.slack.com/methods/conversations.history$pagination
                result = self.client.conversations_history(
                    channel=channel_id,
                    cursor=next_cursor,
                )
                conversation_history = result["messages"]
                # Print results
                logger.info(
                    "{} messages found in {}".format(
                        len(conversation_history), id
                    )
                )
                result_messages.extend(
                    self._read_message(channel_id, message["ts"])
                    for message in conversation_history
                )
                if not result["has_more"]:
                    break
                next_cursor = result["response_metadata"]["next_cursor"]

            except SlackApiError as e:
                if e.response["error"] == "ratelimited":
                    logger.error(
                        "Rate limit error reached, sleeping for: {} seconds".format(
                            e.response.headers["retry-after"]
                        )
                    )
                    time.sleep(int(e.response.headers["retry-after"]))
                else:
                    logger.error(
                        "Error parsing conversation replies: {}".format(e)
                    )

        return (
            "\n\n".join(result_messages)
            if reverse_chronological
            else "\n\n".join(result_messages[::-1])
        )

    def load_data(
        self, channel_ids: List[str], reverse_chronological: bool = True
    ) -> List[Document]:
        """Load data from the input directory.
        Args:
            channel_ids (List[str]): List of channel ids to read.
        Returns:
            List[Document]: List of documents.
        """
        results = []
        for channel_id in channel_ids:
            channel_content = self._read_channel(
                channel_id, reverse_chronological=reverse_chronological
            )
            results.append(
                Document(channel_content, extra_info={"channel": channel_id})
            )
        return results
