"""
Materializer for RawConversationData with a lightweight, styled HTML
visualization.

Design notes
------------
* Re-uses the base CSS from HTMLVisualizer to keep the design consistent
  across all visual artifacts in the project.
* Collapsible <details> sections avoid producing a huge, unreadable page
  when many conversations or messages are present.
* Only the first `MESSAGE_PREVIEW_LIMIT` messages of every conversation are
  rendered; this prevents extremely large artifacts while still providing
  a representative sample.  Adjust the constant if needed.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer

from src.utils.html_visualization import HTMLVisualizer
from src.utils.models import ChatMessage, ConversationData, RawConversationData


class RawConversationDataMaterializer(PydanticMaterializer):
    """Materializer that stores RawConversationData and generates HTML."""

    # ZenML discovery
    ASSOCIATED_TYPES = (RawConversationData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    # Limit messages shown per conversation in the visualization
    MESSAGE_PREVIEW_LIMIT: int = 15

    # --------------------------------------------------------------------- #
    # ZenML hooks                                                           #
    # --------------------------------------------------------------------- #
    def save_visualizations(
        self, data: RawConversationData
    ) -> Dict[str, VisualizationType]:
        """Generate and save HTML visualization for raw conversations.

        Args:
            data: The `RawConversationData` artifact to visualize.

        Returns:
            Mapping of output file path to ZenML `VisualizationType`.
        """
        html = self._build_html(data)
        output_path = self._get_output_path()

        # Write artifact
        with fileio.open(output_path, "w") as f:
            f.write(html)

        # Return mapping understood by ZenML
        return {output_path: VisualizationType.HTML}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _get_output_path(self) -> str:
        """Return absolute path to the rendered HTML inside artifact URI."""
        # self.uri is provided by PydanticMaterializer base class
        return os.path.join(self.uri, "raw_conversations.html")

    # ----------------------- HTML generation --------------------------- #
    def _build_html(self, data: RawConversationData) -> str:
        """Assemble full HTML page."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats = self._calculate_stats(data)

        # CSS – pulled from HTMLVisualizer for consistency
        base_css = HTMLVisualizer().base_css  # type: ignore[misc]

        # Build page
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Raw Conversation Data</title>
            <style>{base_css}</style>
        </head>
        <body>
            <div class="container">
                {self._generate_header(ts)}
                {self._generate_stats_section(stats)}
                {self._generate_conversations_section(data.conversations)}
            </div>
        </body>
        </html>
        """

    # ---------------------- Section builders --------------------------- #
    def _generate_header(self, timestamp: str) -> str:
        """Return stylised header."""
        return f"""
        <div class="header">
            <h1>💬 Raw Conversation Data</h1>
            <p class="subtitle">Visualized on {timestamp}</p>
        </div>
        """

    def _generate_stats_section(self, stats: Dict[str, int]) -> str:
        """Return a small stats grid."""
        return f"""
        <div class="grid grid-stats">
            <div class="stat-card">
                <div class="stat-value">{stats['conversations']}</div>
                <div class="stat-label">Conversations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['messages']}</div>
                <div class="stat-label">Messages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['sources']}</div>
                <div class="stat-label">Sources</div>
            </div>
        </div>
        """

    def _generate_conversations_section(
        self, conversations: List[ConversationData]
    ) -> str:
        """Return expandable HTML for each conversation."""
        if not conversations:
            return '<div class="card"><p>No conversations available.</p></div>'

        section_html = ""
        for idx, conv in enumerate(conversations, start=1):
            summary_line = (
                f"{conv.channel_name}"
                + (f" / {conv.thread_name}" if conv.thread_name else "")
                + f" • {conv.total_messages} messages • {conv.source.title()}"
            )
            messages_html = "".join(
                self._format_message(m)
                for m in conv.messages[: self.MESSAGE_PREVIEW_LIMIT]
            )
            hidden_count = conv.total_messages - self.MESSAGE_PREVIEW_LIMIT
            if hidden_count > 0:
                messages_html += (
                    f'<p style="font-style: italic; color: var(--color-text-muted);">'
                    f"... and {hidden_count} more messages not shown</p>"
                )

            section_html += f"""
            <details class="card" style="margin-bottom: var(--spacing-md);">
                <summary style="cursor: pointer; font-weight: 600;">
                    {idx}. {self._escape_html(summary_line)}
                </summary>
                <div style="margin-top: var(--spacing-md);">
                    {messages_html}
                </div>
            </details>
            """

        return section_html

    # --------------------------- Utils --------------------------------- #
    def _calculate_stats(self, data: RawConversationData) -> Dict[str, int]:
        """Aggregate simple statistics."""
        total_messages = sum(c.total_messages for c in data.conversations)
        return {
            "conversations": data.total_conversations,
            "messages": total_messages,
            "sources": len(set(data.sources)),
        }

    def _format_message(self, msg: ChatMessage) -> str:
        """Convert a ChatMessage into styled HTML."""
        timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        content_text = (
            msg.content[:200] + "…" if len(msg.content) > 200 else msg.content
        )
        return f"""
        <div class="conversation-item">
            <div class="conversation-meta">
                <strong>{self._escape_html(msg.author)}</strong>
                <span>{timestamp}</span>
            </div>
            <p style="margin: 0;">{self._escape_html(content_text)}</p>
        </div>
        """

    @staticmethod
    def _escape_html(text: str) -> str:
        """HTML-escape special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )
