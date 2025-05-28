"""Materializer for AnalysisData with viewpoint tension diagrams and reflection insights."""

import os
from typing import Dict

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
                agreement_html = "<div class='agreement-points'><h3>Main Points of Agreement</h3><ul>"
                for point in va.main_points_of_agreement:
                    agreement_html += f"<li>{point}</li>"
                agreement_html += "</ul></div>"

            # Areas of tension
            tensions_html = ""
            if va.areas_of_tension:
                tensions_html = (
                    "<div class='tensions'><h3>Areas of Tension</h3>"
                )
                for tension in va.areas_of_tension:
                    viewpoints_html = ""
                    for perspective, view in tension.viewpoints.items():
                        viewpoints_html += f"""
                        <div class="viewpoint">
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
                <div class='perspective-gaps'>
                    <h3>Perspective Gaps</h3>
                    <p>{va.perspective_gaps}</p>
                </div>
                """

            # Integrative insights
            insights_html = ""
            if va.integrative_insights:
                insights_html = f"""
                <div class='integrative-insights'>
                    <h3>Integrative Insights</h3>
                    <p>{va.integrative_insights}</p>
                </div>
                """

            viewpoint_html = f"""
            <div class="viewpoint-section">
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
                critique_html = "<div class='critique-summary'><h3>Critique Summary</h3><ul>"
                for critique in rm.critique_summary:
                    critique_html += f"<li>{critique}</li>"
                critique_html += "</ul></div>"

            # Additional questions
            questions_html = ""
            if rm.additional_questions_identified:
                questions_html = "<div class='additional-questions'><h3>Additional Questions Identified</h3><ul>"
                for question in rm.additional_questions_identified:
                    questions_html += f"<li>{question}</li>"
                questions_html += "</ul></div>"

            # Searches performed
            searches_html = ""
            if rm.searches_performed:
                searches_html = "<div class='searches-performed'><h3>Searches Performed</h3><ul>"
                for search in rm.searches_performed:
                    searches_html += f"<li>{search}</li>"
                searches_html += "</ul></div>"

            # Error handling
            error_html = ""
            if rm.error:
                error_html = f"""
                <div class='error-message'>
                    <h3>Error Encountered</h3>
                    <p>{rm.error}</p>
                </div>
                """

            reflection_html = f"""
            <div class="reflection-section">
                <h2>Reflection Metadata</h2>
                <div class="improvements-count">
                    <span class="count">{int(rm.improvements_made)}</span>
                    <span class="label">Improvements Made</span>
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
                '<div class="no-analysis">No analysis data available yet</div>'
            )
        else:
            content_html = viewpoint_html + reflection_html

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Data Visualization</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                    color: #333;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                
                .header {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    margin-bottom: 30px;
                    text-align: center;
                }}
                
                h1 {{
                    margin: 0;
                    color: #2c3e50;
                    font-size: 36px;
                }}
                
                .viewpoint-section,
                .reflection-section {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                }}
                
                h2 {{
                    color: #2c3e50;
                    margin: 0 0 25px 0;
                    padding-bottom: 15px;
                    border-bottom: 2px solid #e9ecef;
                }}
                
                h3 {{
                    color: #495057;
                    margin: 25px 0 15px 0;
                }}
                
                h4 {{
                    color: #6c757d;
                    margin: 15px 0 10px 0;
                }}
                
                .agreement-points {{
                    background: #f0f9ff;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-left: 4px solid #2dce89;
                }}
                
                .tensions {{
                    margin-top: 25px;
                }}
                
                .tension-card {{
                    background: #fff5f5;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                    border: 1px solid #ffe0e0;
                }}
                
                .viewpoints-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                
                .viewpoint {{
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    border: 1px solid #e9ecef;
                }}
                
                .viewpoint-label {{
                    font-weight: bold;
                    color: #495057;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    font-size: 12px;
                    letter-spacing: 0.5px;
                }}
                
                .viewpoint-text {{
                    color: #666;
                    line-height: 1.6;
                }}
                
                .perspective-gaps,
                .integrative-insights {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-top: 20px;
                }}
                
                .perspective-gaps {{
                    border-left: 4px solid #ffd600;
                }}
                
                .integrative-insights {{
                    border-left: 4px solid #5e72e4;
                    background: #f0f5ff;
                }}
                
                .improvements-count {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 25px;
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                }}
                
                .improvements-count .count {{
                    display: block;
                    font-size: 48px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                
                .improvements-count .label {{
                    font-size: 16px;
                    opacity: 0.9;
                }}
                
                .critique-summary,
                .additional-questions,
                .searches-performed {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                
                .critique-summary {{
                    border-left: 4px solid #f5365c;
                    background: #fff5f5;
                }}
                
                .additional-questions {{
                    border-left: 4px solid #11cdef;
                    background: #f0fbff;
                }}
                
                .searches-performed {{
                    border-left: 4px solid #2dce89;
                    background: #f0fff5;
                }}
                
                ul {{
                    margin: 10px 0;
                    padding-left: 25px;
                }}
                
                li {{
                    margin: 8px 0;
                    line-height: 1.6;
                }}
                
                .error-message {{
                    background: #ffe0e0;
                    border: 1px solid #ffb0b0;
                    border-radius: 10px;
                    padding: 20px;
                    margin-top: 20px;
                    color: #d32f2f;
                }}
                
                .no-analysis {{
                    text-align: center;
                    color: #999;
                    font-style: italic;
                    padding: 60px;
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                }}
                
                p {{
                    line-height: 1.8;
                    color: #666;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Research Analysis</h1>
                </div>
                
                {content_html}
            </div>
        </body>
        </html>
        """

        return html
