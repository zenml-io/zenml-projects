"""Materializer for QueryContext with interactive mind map visualization."""

import os
from typing import Dict

from utils.css_utils import (
    create_stat_card,
    get_grid_class,
    get_shared_css_tag,
)
from utils.pydantic_models import QueryContext
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class QueryContextMaterializer(PydanticMaterializer):
    """Materializer for QueryContext with mind map visualization."""

    ASSOCIATED_TYPES = (QueryContext,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: QueryContext
    ) -> Dict[str, VisualizationType]:
        """Create and save mind map visualization for the QueryContext.

        Args:
            data: The QueryContext to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualization_path = os.path.join(self.uri, "query_context.html")
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, context: QueryContext) -> str:
        """Generate HTML mind map visualization for the query context.

        Args:
            context: The QueryContext to visualize

        Returns:
            HTML string
        """
        # Create sub-questions HTML
        sub_questions_html = ""
        if context.sub_questions:
            for i, sub_q in enumerate(context.sub_questions, 1):
                sub_questions_html += f"""
                <div class="sub-question-item">
                    <div class="sub-question-number">{i}</div>
                    <div class="sub-question-text">{sub_q}</div>
                </div>
                """
        else:
            sub_questions_html = (
                '<div class="dr-empty">No sub-questions decomposed yet</div>'
            )

        # Format timestamp
        from datetime import datetime

        timestamp = datetime.fromtimestamp(
            context.decomposition_timestamp
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build stats
        stats_html = f"""
        <div class="{get_grid_class("stats")}">
            {create_stat_card(len(context.sub_questions), "Sub-Questions")}
            {create_stat_card(len(context.main_query.split()), "Words in Query")}
            {create_stat_card(sum(len(q.split()) for q in context.sub_questions), "Total Sub-Question Words")}
        </div>
        """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Query Context - {context.main_query[:50]}...</title>
            {get_shared_css_tag()}
            <style>
                /* Component-specific styles for mind map */
                body {{
                    background: linear-gradient(135deg, var(--color-secondary) 0%, var(--color-accent) 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                
                .dr-container {{
                    background: white;
                    border-radius: 20px;
                    box-shadow: var(--shadow-xl);
                    padding: 40px;
                    max-width: 1200px;
                    width: 90%;
                    margin: 20px;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: var(--spacing-xl);
                }}
                
                .timestamp {{
                    color: var(--color-text-secondary);
                    font-size: 0.875rem;
                    margin-top: var(--spacing-sm);
                }}
                
                .mind-map {{
                    position: relative;
                    margin: var(--spacing-xl) 0;
                }}
                
                .main-query-node {{
                    background: linear-gradient(135deg, var(--color-secondary) 0%, var(--color-accent) 100%);
                    color: white;
                    padding: var(--spacing-lg);
                    border-radius: var(--radius-lg);
                    text-align: center;
                    font-size: 1.25rem;
                    font-weight: bold;
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                    margin-bottom: var(--spacing-xl);
                    position: relative;
                }}
                
                .main-query-node::after {{
                    content: '';
                    position: absolute;
                    bottom: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 2px;
                    height: 20px;
                    background: var(--color-secondary);
                }}
                
                .sub-questions-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: var(--spacing-md);
                    position: relative;
                }}
                
                .sub-questions-container::before {{
                    content: '';
                    position: absolute;
                    top: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 80%;
                    height: 2px;
                    background: linear-gradient(90deg, transparent, var(--color-secondary), transparent);
                }}
                
                .sub-question-item {{
                    background: var(--color-bg-secondary);
                    border: 2px solid var(--color-border);
                    border-radius: var(--radius-md);
                    padding: var(--spacing-md);
                    position: relative;
                    transition: var(--transition-base);
                    display: flex;
                    align-items: flex-start;
                    gap: 15px;
                }}
                
                .sub-question-item:hover {{
                    transform: translateY(-5px);
                    box-shadow: var(--shadow-hover);
                    border-color: var(--color-secondary);
                }}
                
                .sub-question-number {{
                    background: var(--color-secondary);
                    color: white;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    flex-shrink: 0;
                }}
                
                .sub-question-text {{
                    color: var(--color-text-primary);
                    line-height: 1.6;
                    flex: 1;
                }}
                
                @media (max-width: 768px) {{
                    .sub-questions-container {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="dr-container">
                <div class="header">
                    <h1>Query Decomposition Mind Map</h1>
                    <div class="timestamp">Created: {timestamp}</div>
                </div>
                
                <div class="mind-map">
                    <div class="main-query-node">
                        {context.main_query}
                    </div>
                    
                    <div class="sub-questions-container">
                        {sub_questions_html}
                    </div>
                </div>
                
                {stats_html}
            </div>
        </body>
        </html>
        """

        return html
