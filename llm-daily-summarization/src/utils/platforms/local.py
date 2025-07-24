"""
Local markdown deliverer.

Writes summaries and tasks to markdown files on the local filesystem.
The root directory can be customised via the ``LOCAL_OUTPUT_DIR`` env var
or by passing a different value to the constructor.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List

from zenml.logger import get_logger

from ..models import DeliveryResult

logger = get_logger(__name__)


class LocalDeliverer:
    """Deliverer that saves content to markdown files locally."""

    def __init__(self, root_dir: str | None = None) -> None:
        # Allow override via env var
        self.root_dir = root_dir or os.getenv(
            "LOCAL_OUTPUT_DIR", "discord_summaries"
        )
        try:
            os.makedirs(self.root_dir, exist_ok=True)
            logger.debug(
                f"LocalDeliverer initialized with root_dir={self.root_dir}"
            )
        except Exception as exc:
            logger.error(
                f"Failed to create output directory '{self.root_dir}': {exc}"
            )
            raise

    # --------------------------------------------------------------------- #
    # Public API (parity with other deliverers)                             #
    # --------------------------------------------------------------------- #
    def deliver_summary(self, summary: Dict[str, Any]) -> DeliveryResult:
        """Write a single summary to a markdown file."""
        filename = self._build_summary_filename(
            summary.get("title", "summary")
        )
        filepath = os.path.join(self.root_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as fp:
                fp.write(self._render_summary_md(summary))

            logger.info(f"Summary saved locally: {filepath}")
            return DeliveryResult(
                target="local",
                success=True,
                delivered_items=[filename],
                failed_items=[],
                delivery_url=os.path.abspath(filepath),
            )
        except Exception as exc:
            logger.error(f"Failed to write summary to '{filepath}': {exc}")
            return DeliveryResult(
                target="local",
                success=False,
                delivered_items=[],
                failed_items=[filename],
                error_message=str(exc),
            )

    def deliver_tasks(self, tasks: List[Dict[str, Any]]) -> DeliveryResult:
        """Write extracted tasks to a markdown file."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        filename = f"tasks_{timestamp}.md"
        filepath = os.path.join(self.root_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as fp:
                fp.write(self._render_tasks_md(tasks))

            logger.info(f"Tasks saved locally: {filepath}")
            return DeliveryResult(
                target="local",
                success=True,
                delivered_items=[filename],
                failed_items=[],
                delivery_url=os.path.abspath(filepath),
            )
        except Exception as exc:
            logger.error(f"Failed to write tasks to '{filepath}': {exc}")
            return DeliveryResult(
                target="local",
                success=False,
                delivered_items=[],
                failed_items=[filename],
                error_message=str(exc),
            )

    # --------------------------------------------------------------------- #
    # Helper functions                                                      #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _slugify(text: str, max_length: int = 50) -> str:
        """Convert text into a safe filename slug."""
        text = re.sub(r"[\\/*?:\"<>|]", "_", text)  # illegal characters
        text = re.sub(r"\s+", "_", text)  # whitespace to underscore
        return text[:max_length].strip("_") or "untitled"

    def _build_summary_filename(self, title: str) -> str:
        """Build a unique file name for a summary markdown."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
        slug = self._slugify(title)
        return f"{slug}_{timestamp}.md"

    @staticmethod
    def _render_summary_md(summary: Dict[str, Any]) -> str:
        """Render summary content as markdown."""
        lines: List[str] = []
        lines.append(f"# {summary.get('title', 'Conversation Summary')}\n")

        if key_points := summary.get("key_points"):
            lines.append("## Key Points")
            lines.extend([f"- {kp}" for kp in key_points])
            lines.append("")

        if content := summary.get("content"):
            lines.append("## Full Summary")
            lines.append(content)
            lines.append("")

        # Optional metadata
        meta_sections = []
        if participants := summary.get("participants"):
            meta_sections.append(
                f"**Participants:** {', '.join(participants)}"
            )
        if topics := summary.get("topics"):
            meta_sections.append(f"**Topics:** {', '.join(topics)}")
        if word_count := summary.get("word_count"):
            meta_sections.append(f"**Word Count:** {word_count}")
        if confidence := summary.get("confidence_score"):
            meta_sections.append(f"**Confidence:** {confidence:.2f}")

        if meta_sections:
            lines.append("## Metadata")
            lines.extend(meta_sections)
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _render_tasks_md(tasks: List[Dict[str, Any]]) -> str:
        """Render tasks as markdown."""
        lines: List[str] = ["# Action Items\n"]
        for idx, task in enumerate(tasks, start=1):
            title = task.get("title", f"Task {idx}")
            desc = task.get("description", "")
            assignee = task.get("assignee") or "Unassigned"
            priority = task.get("priority", "medium").capitalize()
            due = task.get("due_date") or "n/a"
            confidence = task.get("confidence_score", None)

            lines.append(f"## {idx}. {title}")
            if desc:
                lines.append(desc)
            lines.append(f"- Assignee: **{assignee}**")
            lines.append(f"- Priority: **{priority}**")
            lines.append(f"- Due: **{due}**")
            if confidence is not None:
                lines.append(f"- Confidence: **{confidence:.2f}**")
            lines.append("")

        return "\n".join(lines).strip() + "\n"
