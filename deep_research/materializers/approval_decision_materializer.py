"""Materializer for ApprovalDecision with custom visualization."""

import os
from datetime import datetime
from typing import Dict

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

        # Determine status color and icon
        if decision.approved:
            status_color = "#27ae60"
            status_icon = "✅"
            status_text = "APPROVED"
        else:
            status_color = "#e74c3c"
            status_icon = "❌"
            status_text = "NOT APPROVED"

        # Format approval method
        method_display = {
            "APPROVE_ALL": "Approve All Queries",
            "SKIP": "Skip Additional Research",
            "SELECT_SPECIFIC": "Select Specific Queries",
        }.get(decision.approval_method, decision.approval_method or "Unknown")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Approval Decision</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 30px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid {status_color};
                    padding-bottom: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }}
                .status-badge {{
                    background-color: {status_color};
                    color: white;
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 18px;
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }}
                .section {{
                    margin: 25px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .info-card {{
                    background-color: #fff;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
                }}
                .info-label {{
                    color: #7f8c8d;
                    font-size: 14px;
                    margin-bottom: 5px;
                }}
                .info-value {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .selected-queries {{
                    background-color: #e8f8f5;
                    border-left: 4px solid #27ae60;
                }}
                .query-list {{
                    margin: 15px 0;
                }}
                .query-item {{
                    background-color: white;
                    padding: 12px;
                    margin: 8px 0;
                    border-radius: 4px;
                    border-left: 3px solid #3498db;
                    display: flex;
                    align-items: center;
                }}
                .query-number {{
                    background-color: #3498db;
                    color: white;
                    border-radius: 50%;
                    width: 25px;
                    height: 25px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 12px;
                    font-size: 14px;
                    font-weight: bold;
                }}
                .notes-section {{
                    background-color: #fef9e7;
                    border-left: 4px solid #f39c12;
                }}
                .notes-content {{
                    background-color: white;
                    padding: 15px;
                    border-radius: 4px;
                    margin-top: 10px;
                    font-style: italic;
                    color: #555;
                    white-space: pre-wrap;
                }}
                .timestamp {{
                    text-align: right;
                    color: #7f8c8d;
                    font-size: 14px;
                    margin-top: 20px;
                    padding-top: 15px;
                    border-top: 1px dashed #ddd;
                }}
                .icon {{
                    margin-right: 8px;
                }}
                .empty-state {{
                    color: #95a5a6;
                    font-style: italic;
                    text-align: center;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>
                    <span>🔒 Approval Decision</span>
                    <div class="status-badge">
                        <span>{status_icon}</span>
                        <span>{status_text}</span>
                    </div>
                </h1>
                
                <div class="info-grid">
                    <div class="info-card">
                        <div class="info-label">Approval Method</div>
                        <div class="info-value">{method_display}</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">Decision Time</div>
                        <div class="info-value">{decision_time}</div>
                    </div>
                    <div class="info-card">
                        <div class="info-label">Queries Selected</div>
                        <div class="info-value">{len(decision.selected_queries)}</div>
                    </div>
                </div>
        """

        # Add selected queries section if any
        if decision.selected_queries:
            html += """
                <div class="section selected-queries">
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
            html += """
                <div class="section">
                    <h2><span class="icon">📋</span>Selected Queries</h2>
                    <div class="empty-state">
                        No queries were selected for additional research
                    </div>
                </div>
            """

        # Add reviewer notes if any
        if decision.reviewer_notes:
            html += f"""
                <div class="section notes-section">
                    <h2><span class="icon">📝</span>Reviewer Notes</h2>
                    <div class="notes-content">
                        {decision.reviewer_notes}
                    </div>
                </div>
            """

        # Add timestamp footer
        html += f"""
                <div class="timestamp">
                    <strong>Decision recorded at:</strong> {decision_time}
                </div>
            </div>
        </body>
        </html>
        """

        return html
