"""Materializer for AnalysisData with viewpoint tension diagrams and reflection insights."""

import os
from typing import Dict

from utils.css_utils import (
    get_card_class,
    get_section_class,
    get_shared_css_tag,
)
from utils.pydantic_models import AnalysisData
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class AnalysisDataMaterializer(PydanticMaterializer):
    """Materializer for AnalysisData with viewpoint and reflection visualization."""

    ASSOCIATED_TYPES = (AnalysisData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: AnalysisData
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the AnalysisData.

        Args:
            data: The AnalysisData to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualization_path = os.path.join(self.uri, "analysis_data.html")
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, data: AnalysisData) -> str:
        """Generate HTML visualization for the analysis data.

        Args:
            data: The AnalysisData to visualize

        Returns:
            HTML string
        """
        # Viewpoint analysis section
        viewpoint_html = ""
        if data.viewpoint_analysis:
            va = data.viewpoint_analysis

            # Points of agreement
            agreement_html = ""
            if va.main_points_of_agreement:
                agreement_html = "<div class='dr-section dr-section--success'><h3>Main Points of Agreement</h3><ul class='dr-list'>"
                for point in va.main_points_of_agreement:
                    agreement_html += f"<li>{point}</li>"
                agreement_html += "</ul></div>"

            # Areas of tension
            tensions_html = ""
            if va.areas_of_tension:
                tensions_html = (
                    "<div class='tensions-section'><h3>Areas of Tension</h3>"
                )
                for tension in va.areas_of_tension:
                    viewpoints_html = ""
                    for perspective, view in tension.viewpoints.items():
                        viewpoints_html += f"""
                        <div class="viewpoint-item">
                            <div class="viewpoint-label">{perspective}</div>
                            <div class="viewpoint-text">{view}</div>
                        </div>
                        """

                    tensions_html += f"""
                    <div class="tension-card">
                        <h4>{tension.topic}</h4>
                        <div class="viewpoints-container">
                            {viewpoints_html}
                        </div>
                    </div>
                    """
                tensions_html += "</div>"

            # Perspective gaps
            gaps_html = ""
            if va.perspective_gaps:
                gaps_html = f"""
                <div class='{get_section_class("warning")}'>
                    <h3>Perspective Gaps</h3>
                    <p>{va.perspective_gaps}</p>
                </div>
                """

            # Integrative insights
            insights_html = ""
            if va.integrative_insights:
                insights_html = f"""
                <div class='{get_section_class("info")}'>
                    <h3>Integrative Insights</h3>
                    <p>{va.integrative_insights}</p>
                </div>
                """

            viewpoint_html = f"""
            <div class="{get_card_class()}">
                <h2>Viewpoint Analysis</h2>
                {agreement_html}
                {tensions_html}
                {gaps_html}
                {insights_html}
            </div>
            """

        # Reflection metadata section
        reflection_html = ""
        if data.reflection_metadata:
            rm = data.reflection_metadata

            # Critique summary
            critique_html = ""
            if rm.critique_summary:
                critique_html = "<div class='dr-section dr-section--danger'><h3>Critique Summary</h3><ul class='dr-list'>"
                for critique in rm.critique_summary:
                    critique_html += f"<li>{critique}</li>"
                critique_html += "</ul></div>"

            # Additional questions
            questions_html = ""
            if rm.additional_questions_identified:
                questions_html = "<div class='dr-section dr-section--info'><h3>Additional Questions Identified</h3><ul class='dr-list'>"
                for question in rm.additional_questions_identified:
                    questions_html += f"<li>{question}</li>"
                questions_html += "</ul></div>"

            # Searches performed
            searches_html = ""
            if rm.searches_performed:
                searches_html = "<div class='dr-section dr-section--success'><h3>Searches Performed</h3><ul class='dr-list'>"
                for search in rm.searches_performed:
                    searches_html += f"<li>{search}</li>"
                searches_html += "</ul></div>"

            # Error handling
            error_html = ""
            if rm.error:
                error_html = f"""
                <div class='dr-notice dr-notice--warning'>
                    <h3>Error Encountered</h3>
                    <p>{rm.error}</p>
                </div>
                """

            reflection_html = f"""
            <div class="{get_card_class()}">
                <h2>Reflection Metadata</h2>
                <div class="improvements-counter">
                    <span class="count-value">{int(rm.improvements_made)}</span>
                    <span class="count-label">Improvements Made</span>
                </div>
                {critique_html}
                {questions_html}
                {searches_html}
                {error_html}
            </div>
            """

        # Handle empty state
        if not viewpoint_html and not reflection_html:
            content_html = (
                '<div class="dr-empty">No analysis data available yet</div>'
            )
        else:
            content_html = viewpoint_html + reflection_html

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Data Visualization</title>
            {get_shared_css_tag()}
            <style>
                /* Component-specific styles */
                body {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                }}
                
                .tensions-section {{
                    margin-top: var(--spacing-lg);
                }}
                
                .tension-card {{
                    background: #fff5f5;
                    border-radius: var(--radius-md);
                    padding: var(--spacing-md);
                    margin-bottom: var(--spacing-md);
                    border: 1px solid #ffe0e0;
                }}
                
                .viewpoints-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                
                .viewpoint-item {{
                    background: var(--color-bg-secondary);
                    border-radius: var(--radius-md);
                    padding: 15px;
                    border: 1px solid var(--color-border);
                }}
                
                .viewpoint-label {{
                    font-weight: bold;
                    color: var(--color-text-secondary);
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    font-size: 0.75rem;
                    letter-spacing: 0.5px;
                }}
                
                .viewpoint-text {{
                    color: var(--color-text-secondary);
                    line-height: 1.6;
                }}
                
                .improvements-counter {{
                    background: linear-gradient(135deg, var(--color-secondary) 0%, var(--color-accent) 100%);
                    color: white;
                    border-radius: var(--radius-md);
                    padding: var(--spacing-md);
                    text-align: center;
                    margin-bottom: var(--spacing-lg);
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                }}
                
                .count-value {{
                    display: block;
                    font-size: 3rem;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                
                .count-label {{
                    font-size: 1rem;
                    opacity: 0.9;
                }}
            </style>
        </head>
        <body>
            <div class="dr-container">
                <div class="dr-header-card dr-text-center">
                    <h1>Research Analysis</h1>
                </div>
                
                {content_html}
            </div>
        </body>
        </html>
        """

        return html
