"""Materializer for ApprovalDecision with custom visualization."""

import os
from datetime import datetime
from typing import Dict

from utils.css_utils import (
    get_card_class,
    get_grid_class,
    get_section_class,
    get_shared_css_tag,
)
from utils.pydantic_models import ApprovalDecision
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class ApprovalDecisionMaterializer(PydanticMaterializer):
    """Materializer for the ApprovalDecision class with visualizations."""

    ASSOCIATED_TYPES = (ApprovalDecision,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: ApprovalDecision
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the ApprovalDecision.

        Args:
            data: The ApprovalDecision to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization
        visualization_path = os.path.join(self.uri, "approval_decision.html")

        # Create HTML content
        html_content = self._generate_visualization_html(data)

        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, decision: ApprovalDecision) -> str:
        """Generate HTML visualization for the approval decision.

        Args:
            decision: The ApprovalDecision to visualize

        Returns:
            HTML string
        """
        # Format timestamp
        decision_time = datetime.fromtimestamp(decision.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Determine status icon and text
        if decision.approved:
            status_icon = "✅"
            status_text = "APPROVED"
        else:
            status_icon = "❌"
            status_text = "NOT APPROVED"

        # Format approval method
        method_display = {
            "APPROVE_ALL": "Approve All Queries",
            "SKIP": "Skip Additional Research",
            "SELECT_SPECIFIC": "Select Specific Queries",
        }.get(decision.approval_method, decision.approval_method or "Unknown")

        # Build info cards
        info_cards_html = f"""
        <div class="{get_grid_class("cards")}">
            <div class="dr-card dr-card--bordered">
                <div class="info-label">Approval Method</div>
                <div class="info-value">{method_display}</div>
            </div>
            <div class="dr-card dr-card--bordered">
                <div class="info-label">Decision Time</div>
                <div class="info-value">{decision_time}</div>
            </div>
            <div class="dr-card dr-card--bordered">
                <div class="info-label">Queries Selected</div>
                <div class="info-value">{len(decision.selected_queries)}</div>
            </div>
        </div>
        """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Approval Decision</title>
            {get_shared_css_tag()}
            <style>
                /* Component-specific styles */
                .status-badge {{
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 1.125rem;
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                
                .status-approved {{
                    background-color: var(--color-success);
                    color: white;
                }}
                
                .status-rejected {{
                    background-color: var(--color-danger);
                    color: white;
                }}
                
                .info-label {{
                    color: var(--color-text-light);
                    font-size: 0.875rem;
                    margin-bottom: 5px;
                }}
                
                .info-value {{
                    font-size: 1.125rem;
                    font-weight: bold;
                    color: var(--color-heading);
                }}
                
                .query-item {{
                    background-color: white;
                    padding: 12px;
                    margin: 8px 0;
                    border-radius: var(--radius-sm);
                    border-left: 3px solid var(--color-primary);
                    display: flex;
                    align-items: center;
                }}
                
                .query-number {{
                    background-color: var(--color-primary);
                    color: white;
                    border-radius: 50%;
                    width: 25px;
                    height: 25px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 12px;
                    font-size: 0.875rem;
                    font-weight: bold;
                }}
                
                .notes-content {{
                    background-color: white;
                    padding: 15px;
                    border-radius: var(--radius-sm);
                    margin-top: 10px;
                    font-style: italic;
                    color: var(--color-text-secondary);
                    white-space: pre-wrap;
                }}
                
                .icon {{
                    margin-right: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="dr-container dr-container--narrow">
                <div class="{get_card_class()}">
                    <h1 class="dr-flex-between">
                        <span>🔒 Approval Decision</span>
                        <div class="status-badge status-{"approved" if decision.approved else "rejected"}">
                            <span>{status_icon}</span>
                            <span>{status_text}</span>
                        </div>
                    </h1>
                    
                    {info_cards_html}
        """

        # Add selected queries section if any
        if decision.selected_queries:
            html += f"""
                <div class="{get_section_class("success")}">
                    <h2><span class="icon">📋</span>Selected Queries</h2>
                    <div class="query-list">
            """

            for i, query in enumerate(decision.selected_queries, 1):
                html += f"""
                        <div class="query-item">
                            <div class="query-number">{i}</div>
                            <div>{query}</div>
                        </div>
                """

            html += """
                    </div>
                </div>
            """
        else:
            html += f"""
                <div class="{get_section_class()}">
                    <h2><span class="icon">📋</span>Selected Queries</h2>
                    <div class="dr-empty dr-text-center">
                        No queries were selected for additional research
                    </div>
                </div>
            """

        # Add reviewer notes if any
        if decision.reviewer_notes:
            html += f"""
                <div class="{get_section_class("warning")}">
                    <h2><span class="icon">📝</span>Reviewer Notes</h2>
                    <div class="notes-content">
                        {decision.reviewer_notes}
                    </div>
                </div>
            """

        # Add timestamp footer
        html += f"""
                    <div class="dr-timestamp">
                        <strong>Decision recorded at:</strong> {decision_time}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html
