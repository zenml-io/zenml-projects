"""
Discord platform helpers.

Contains:
    • DiscordDeliverer – posts summaries and tasks to a Discord channel.

Note: The fetch client (`DiscordClient`) remains in utils.chat_clients to
avoid a large diff; we import it from there.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import discord
from zenml.logger import get_logger

from ..models import DeliveryResult

logger = get_logger(__name__)


class DiscordDeliverer:
    """Delivers content to Discord channels."""

    def __init__(
        self,
        token: str,
        channel_id: str,
        summary_config: Dict[str, Any] | None = None,
    ):
        """
        Args:
            token: Discord bot token
            channel_id: Discord channel ID for posting summaries
            summary_config: Optional formatting configuration
        """
        self.token = token
        self.channel_id = channel_id
        self.summary_config = summary_config or {}

    async def _post_message_chunks(
        self, channel_id: str, text: str, *, max_length: int = 1900
    ) -> bool:
        """
        Post text to Discord, splitting into chunks that respect Discord's
        2 000-character limit (we default to 1 900 to keep margin).

        Returns:
            True on success, False otherwise.
        """
        intents = discord.Intents.default()
        intents.guilds = True
        client = discord.Client(intents=intents)

        success = False

        @client.event
        async def on_ready():
            nonlocal success
            try:
                channel = await client.fetch_channel(int(channel_id))

                chunks = self._split_message_for_discord(text, max_length)
                for idx, chunk in enumerate(chunks):
                    if idx:  # polite rate-limit
                        await asyncio.sleep(1)
                    await channel.send(chunk, suppress_embeds=True)
                    logger.info(
                        "Posted chunk %d/%d to Discord channel %s",
                        idx + 1,
                        len(chunks),
                        channel_id,
                    )
                success = True
            except Exception as exc:  # pragma: no cover
                logger.error("Error posting to Discord: %s", exc)
            finally:
                await client.close()
                await asyncio.sleep(0.1)

        try:
            await client.start(self.token)
        except Exception as exc:  # pragma: no cover
            logger.error("Discord client error: %s", exc)
        finally:
            if not client.is_closed():
                await client.close()

        return success

    @staticmethod
    def _split_message_for_discord(text: str, max_length: int) -> List[str]:
        """
        Split long text into chunks that fit within Discord's message limit.

        The algorithm keeps section headers (`## `) and bullet lists together
        when possible for readability.
        """
        if len(text) <= max_length:
            return [text]

        chunks: List[str] = []
        lines = text.split("\n")

        sections: List[str] = []
        current_section: List[str] = []

        for line in lines:
            if line.startswith("## ") and current_section:
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        if current_section:
            sections.append("\n".join(current_section))

        current_chunk = sections[0] if sections else ""

        for section in sections[1:]:
            if len(current_chunk) + len(section) + 2 > max_length:
                if len(section) > max_length:
                    if current_chunk.strip():
                        chunks.append(current_chunk)
                    section_lines = section.split("\n")
                    current_chunk = section_lines[0] + "\n"
                    for ln in section_lines[1:]:
                        if len(current_chunk) + len(ln) + 1 <= max_length:
                            current_chunk += ln + "\n"
                        else:
                            chunks.append(current_chunk.rstrip())
                            current_chunk = ln + "\n"
                else:
                    chunks.append(current_chunk)
                    current_chunk = section
            else:
                current_chunk += "\n\n" + section

        if current_chunk.strip():
            chunks.append(current_chunk)

        return [c for c in chunks if c.strip() and len(c.strip()) > 10]

    # ---------------------------------------------------------------------#
    # Internal helper: consolidated digest formatting                      #
    # ---------------------------------------------------------------------#
    def _format_consolidated_for_discord(
        self,
        summaries: List[Dict[str, Any]],
        tasks: Optional[List[Dict[str, Any]]] | None = None,
    ) -> str:
        """Build one Discord-friendly digest covering all summaries (+ tasks).

        Confidence scores and word counts are deliberately excluded.
        """
        if not summaries:
            return "No conversation summaries available today."

        lines: List[str] = []
        lines.append(
            f"📰 **Daily Conversation Digest – {len(summaries)} summaries**\n"
        )

        # ------------------------------------------------------------------#
        # Summaries section                                                 #
        # ------------------------------------------------------------------#
        for idx, summary in enumerate(summaries, 1):
            title = summary.get("title", f"Conversation {idx}")
            lines.append(f"## {idx}. **{title}**\n")

            # Key Points
            if summary.get("key_points") and self.summary_config.get(
                "include_key_points", True
            ):
                lines.append("**Key Points:**")
                max_points = self.summary_config.get("max_key_points", 5)
                for point in summary["key_points"][:max_points]:
                    lines.append(f"• {point}")
                lines.append("")  # blank line

            # Content (optionally truncated)
            if self.summary_config.get(
                "include_content", True
            ) and summary.get("content"):
                content = summary["content"]
                max_content = self.summary_config.get(
                    "max_content_length", 800
                )
                if len(content) > max_content:
                    content = f"{content[: max_content - 3]}..."
                lines.append(content)
                lines.append("")

            # Metadata (participants / topics only)
            meta_parts: List[str] = []
            if summary.get("participants") and self.summary_config.get(
                "show_participants", True
            ):
                participants = ", ".join(summary["participants"][:5])
                if len(summary["participants"]) > 5:
                    participants += (
                        f" (+{len(summary['participants']) - 5} more)"
                    )
                meta_parts.append(f"👥 **Participants:** {participants}")

            if summary.get("topics") and self.summary_config.get(
                "show_topics", True
            ):
                topics = ", ".join(summary["topics"][:3])
                meta_parts.append(f"🏷️ **Topics:** {topics}")

            if meta_parts:
                lines.extend(meta_parts)
                lines.append("")

        # ------------------------------------------------------------------#
        # Tasks section (optional)                                          #
        # ------------------------------------------------------------------#
        if tasks:
            lines.append(f"✅ **Action Items ({len(tasks)} tasks)**\n")
            max_tasks = self.summary_config.get("max_tasks_display", 10)
            for t_idx, task in enumerate(tasks[:max_tasks], 1):
                priority_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢",
                }.get(task.get("priority", "medium"), "⚪")
                lines.append(
                    f"{t_idx}. {priority_emoji} **{task.get('title','Task')}**"
                )

                if self.summary_config.get("include_task_details", True):
                    # Description
                    desc = task.get("description", "")
                    max_desc = self.summary_config.get(
                        "max_task_description_length", 100
                    )
                    if desc:
                        if len(desc) > max_desc:
                            desc = f"{desc[: max_desc - 3]}..."
                        lines.append(f"   {desc}")

                    # Assignee
                    if task.get("assignee") and self.summary_config.get(
                        "show_assignees", True
                    ):
                        lines.append(f"   👤 Assigned to: {task['assignee']}")

                    # Due date
                    if task.get("due_date") and self.summary_config.get(
                        "show_due_dates", False
                    ):
                        lines.append(f"   📅 Due: {task['due_date']}")

                lines.append("")  # blank line between tasks

            if len(tasks) > max_tasks:
                lines.append(f"...and {len(tasks) - max_tasks} more tasks\n")

        return "\n".join(lines).strip()

    # ---------------------------------------------------------------------#
    # Public API – summary and task delivery                               #
    # ---------------------------------------------------------------------#
    async def deliver_summary(
        self, summary_data: Dict[str, Any]
    ) -> DeliveryResult:
        """Deliver a conversation summary to Discord."""
        try:
            logger.info(
                "Delivering summary to Discord channel %s: %s",
                self.channel_id,
                summary_data["title"],
            )

            # Format and maybe truncate the message
            discord_message = self._format_summary_for_discord(summary_data)

            max_length = self.summary_config.get("max_length", 1900)
            if (
                self.summary_config.get("truncate_to_limit", False)
                and len(discord_message) > max_length
            ):
                discord_message = f"{discord_message[: max_length - 3]}..."

            success = await self._post_message_chunks(
                self.channel_id, discord_message, max_length=max_length
            )

            return DeliveryResult(
                target="discord",
                success=success,
                delivered_items=[summary_data["title"]] if success else [],
                failed_items=[] if success else [summary_data["title"]],
                delivery_url=(
                    f"https://discord.com/channels/@me/{self.channel_id}"
                    if success
                    else None
                ),
                error_message=None
                if success
                else "Failed to post to Discord channel",
            )
        except Exception as exc:
            logger.error("Failed to deliver summary to Discord: %s", exc)
            return DeliveryResult(
                target="discord",
                success=False,
                delivered_items=[],
                failed_items=[summary_data["title"]],
                delivery_url=None,
                error_message=str(exc),
            )

    async def deliver_tasks(
        self, tasks: List[Dict[str, Any]]
    ) -> DeliveryResult:
        """Deliver task list to Discord."""
        try:
            logger.info(
                "Delivering %d tasks to Discord channel %s",
                len(tasks),
                self.channel_id,
            )

            discord_message = self._format_tasks_for_discord(tasks)
            max_length = self.summary_config.get("max_length", 1900)

            success = await self._post_message_chunks(
                self.channel_id, discord_message, max_length=max_length
            )

            return DeliveryResult(
                target="discord",
                success=success,
                delivered_items=[t["title"] for t in tasks] if success else [],
                failed_items=[] if success else [t["title"] for t in tasks],
                delivery_url=(
                    f"https://discord.com/channels/@me/{self.channel_id}"
                    if success
                    else None
                ),
                error_message=None
                if success
                else "Failed to post tasks to Discord channel",
            )
        except Exception as exc:
            logger.error("Failed to deliver tasks to Discord: %s", exc)
            return DeliveryResult(
                target="discord",
                success=False,
                delivered_items=[],
                failed_items=[t["title"] for t in tasks],
                delivery_url=None,
                error_message=str(exc),
            )

    async def deliver_consolidated(
        self,
        summaries: List[Dict[str, Any]],
        tasks: Optional[List[Dict[str, Any]]] | None = None,
    ) -> DeliveryResult:
        """Deliver a single consolidated digest to Discord.

        Args:
            summaries: List of summary dicts.
            tasks: Optional list of task dicts to include.

        Returns:
            DeliveryResult indicating success / failure.
        """
        try:
            logger.info(
                "Delivering consolidated digest (%d summaries%s) to Discord channel %s",
                len(summaries),
                f", {len(tasks)} tasks" if tasks else "",
                self.channel_id,
            )

            discord_message = self._format_consolidated_for_discord(
                summaries, tasks
            )
            max_length = self.summary_config.get("max_length", 1900)

            success = await self._post_message_chunks(
                self.channel_id, discord_message, max_length=max_length
            )

            delivered_titles = [s.get("title", "Summary") for s in summaries]
            if tasks:
                delivered_titles.extend(
                    [t.get("title", "Task") for t in tasks]
                )

            return DeliveryResult(
                target="discord",
                success=success,
                delivered_items=delivered_titles if success else [],
                failed_items=[] if success else delivered_titles,
                delivery_url=(
                    f"https://discord.com/channels/@me/{self.channel_id}"
                    if success
                    else None
                ),
                error_message=None
                if success
                else "Failed to post consolidated digest to Discord channel",
            )
        except Exception as exc:
            logger.error(
                "Failed to deliver consolidated digest to Discord: %s", exc
            )
            delivered_titles = [s.get("title", "Summary") for s in summaries]
            if tasks:
                delivered_titles.extend(
                    [t.get("title", "Task") for t in tasks]
                )
            return DeliveryResult(
                target="discord",
                success=False,
                delivered_items=[],
                failed_items=delivered_titles,
                delivery_url=None,
                error_message=str(exc),
            )

    # ---------------------------------------------------------------------#
    # Internal helpers (identical to previous implementation)              #
    # ---------------------------------------------------------------------#
    def _format_summary_for_discord(self, summary_data: Dict[str, Any]) -> str:
        """Format summary based on config (compact / full)."""
        if self.summary_config.get("format", "full") == "compact":
            message = f"📝 **{summary_data['title']}**\n\n"
            if summary_data.get("key_points"):
                message += "**Key Points:**\n"
                max_points = self.summary_config.get("max_key_points", 5)
                for point in summary_data["key_points"][:max_points]:
                    message += f"• {point}\n"
            return message

        # Full format
        message = f"📝 **{summary_data['title']}**\n\n"
        if self.summary_config.get("include_content", True):
            content = summary_data["content"]
            max_content = self.summary_config.get("max_content_length", 800)
            if len(content) > max_content:
                content = f"{content[: max_content - 3]}..."
            message += f"{content}\n\n"

        if summary_data.get("key_points") and self.summary_config.get(
            "include_key_points", True
        ):
            message += "**Key Points:**\n"
            max_points = self.summary_config.get("max_key_points", 5)
            for point in summary_data["key_points"][:max_points]:
                message += f"• {point}\n"
            message += "\n"

        if self.summary_config.get("include_metadata", True):
            parts: List[str] = []
            if summary_data.get("participants") and self.summary_config.get(
                "show_participants", True
            ):
                participants = ", ".join(summary_data["participants"][:5])
                if len(summary_data["participants"]) > 5:
                    participants += (
                        f" (+{len(summary_data['participants']) - 5} more)"
                    )
                parts.append(f"👥 **Participants:** {participants}")

            if summary_data.get("topics") and self.summary_config.get(
                "show_topics", True
            ):
                topics = ", ".join(summary_data["topics"][:3])
                parts.append(f"🏷️ **Topics:** {topics}")

            if summary_data.get("word_count") and self.summary_config.get(
                "show_stats", False
            ):
                parts.append(f"📊 **Words:** {summary_data['word_count']}")

            if parts:
                message += "\n".join(parts) + "\n"

        return message.strip()

    def _format_tasks_for_discord(self, tasks: List[Dict[str, Any]]) -> str:
        """Create a nicely formatted task list."""
        if not tasks:
            return "No action items identified today."

        message = f"✅ **Action Items ({len(tasks)} tasks)**\n\n"
        max_tasks = self.summary_config.get("max_tasks_display", 10)
        shown = tasks[:max_tasks]

        for idx, task in enumerate(shown, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                task.get("priority", "medium"), "⚪"
            )
            message += f"{idx}. {priority_emoji} **{task['title']}**\n"

            if self.summary_config.get("include_task_details", True):
                desc = task["description"]
                max_desc = self.summary_config.get(
                    "max_task_description_length", 100
                )
                if len(desc) > max_desc:
                    desc = f"{desc[: max_desc - 3]}..."
                message += f"   {desc}\n"

                if task.get("assignee") and self.summary_config.get(
                    "show_assignees", True
                ):
                    message += f"   👤 Assigned to: {task['assignee']}\n"

                if task.get("due_date") and self.summary_config.get(
                    "show_due_dates", False
                ):
                    message += f"   📅 Due: {task['due_date']}\n"

            message += "\n"

        if len(tasks) > max_tasks:
            message += f"... and {len(tasks) - max_tasks} more tasks\n"

        return message.strip()
