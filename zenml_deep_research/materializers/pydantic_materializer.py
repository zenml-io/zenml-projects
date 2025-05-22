"""Pydantic materializer for research state objects.

This module contains an extended version of ZenML's PydanticMaterializer
that adds visualization capabilities for the ResearchState model.
"""

import os
from typing import Dict

from utils.pydantic_models import ResearchState
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class ResearchStateMaterializer(PydanticMaterializer):
    """Materializer for the ResearchState class with visualizations."""

    ASSOCIATED_TYPES = (ResearchState,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: ResearchState
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the ResearchState.

        Args:
            data: The ResearchState to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization
        visualization_path = os.path.join(self.uri, "research_state.html")

        # Create HTML content based on current stage
        html_content = self._generate_visualization_html(data)

        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, state: ResearchState) -> str:
        """Generate HTML visualization for the research state.

        Args:
            state: The ResearchState to visualize

        Returns:
            HTML string
        """
        # Base structure for the HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research State: {state.main_query}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .stages {{
                    display: flex;
                    justify-content: space-between;
                    margin: 20px 0;
                    overflow-x: auto;
                }}
                .stage {{
                    padding: 10px 15px;
                    border-radius: 20px;
                    background-color: #e0e0e0;
                    margin-right: 5px;
                    font-size: 14px;
                    white-space: nowrap;
                }}
                .stage.active {{
                    background-color: #2ecc71;
                    color: white;
                    font-weight: bold;
                }}
                .query {{
                    background-color: #eef2f7;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                .sub-questions {{
                    margin: 15px 0;
                }}
                .sub-question {{
                    background-color: #f1f8ff;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 4px;
                }}
                .search-result {{
                    border-left: 3px solid #f39c12;
                    padding: 10px;
                    margin: 10px 0;
                    background-color: #fef9e7;
                }}
                .synthesized {{
                    border-left: 3px solid #27ae60;
                    padding: 10px;
                    margin: 10px 0;
                    background-color: #e8f8f5;
                }}
                .viewpoints {{
                    border-left: 3px solid #8e44ad;
                    padding: 10px;
                    margin: 10px 0;
                    background-color: #f5eef8;
                }}
                .reflection {{
                    border-left: 3px solid #e74c3c;
                    padding: 10px;
                    margin: 10px 0;
                    background-color: #fdedec;
                }}
                .confidence {{
                    display: inline-block;
                    padding: 3px 8px;
                    border-radius: 3px;
                    font-size: 12px;
                    margin-left: 10px;
                }}
                .high {{
                    background-color: #d4edda;
                    color: #155724;
                }}
                .medium {{
                    background-color: #fff3cd;
                    color: #856404;
                }}
                .low {{
                    background-color: #f8d7da;
                    color: #721c24;
                }}
                pre {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 14px;
                }}
                .metadata {{
                    font-size: 12px;
                    color: #7f8c8d;
                    margin-top: 10px;
                    padding-top: 5px;
                    border-top: 1px dashed #ddd;
                }}
                .sources {{
                    margin-top: 10px;
                    font-size: 12px;
                }}
                .sources a {{
                    color: #3498db;
                    text-decoration: none;
                }}
                .sources a:hover {{
                    text-decoration: underline;
                }}
                .progress {{
                    margin-top: 15px;
                    background-color: #ecf0f1;
                    border-radius: 10px;
                    height: 10px;
                    overflow: hidden;
                }}
                .progress-bar {{
                    height: 100%;
                    background-color: #3498db;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Research State</h1>
                
                <!-- Progress stages -->
                <div class="stages">
                    <div class="stage {self._get_stage_class(state, "initial")}">Initial Query</div>
                    <div class="stage {self._get_stage_class(state, "after_query_decomposition")}">Query Decomposition</div>
                    <div class="stage {self._get_stage_class(state, "after_search")}">Information Gathering</div>
                    <div class="stage {self._get_stage_class(state, "after_synthesis")}">Information Synthesis</div>
                    <div class="stage {self._get_stage_class(state, "after_viewpoint_analysis")}">Viewpoint Analysis</div>
                    <div class="stage {self._get_stage_class(state, "after_reflection")}">Reflection & Enhancement</div>
                    <div class="stage {self._get_stage_class(state, "final_report")}">Final Report</div>
                </div>
                
                <!-- Overall progress bar -->
                <div class="progress">
                    <div class="progress-bar" style="width: {self._calculate_progress(state)}%"></div>
                </div>
        """

        # Add main query section
        if state.main_query:
            html += f"""
                <div class="section">
                    <h2>Main Query</h2>
                    <div class="query">
                        <p>{state.main_query}</p>
                    </div>
                </div>
            """

        # Add sub-questions section
        if state.sub_questions:
            html += f"""
                <div class="section">
                    <h2>Sub-Questions ({len(state.sub_questions)})</h2>
                    <div class="sub-questions">
            """

            for i, question in enumerate(state.sub_questions):
                html += f"""
                        <div class="sub-question">
                            <strong>{i + 1}.</strong> {question}
                        </div>
                """

            html += """
                    </div>
                </div>
            """

        # Add search results if available
        if state.search_results:
            html += f"""
                <div class="section">
                    <h2>Search Results</h2>
            """

            for question, results in state.search_results.items():
                html += f"""
                    <h3>{question}</h3>
                    <p>Found {len(results)} results</p>
                    <ul>
                """

                for result in results:
                    # Extract domain from URL
                    domain = ""
                    try:
                        from urllib.parse import urlparse

                        parsed_url = urlparse(result.url)
                        domain = parsed_url.netloc
                        # Strip www. prefix to save space
                        if domain.startswith("www."):
                            domain = domain[4:]
                    except:
                        domain = (
                            result.url.split("/")[2]
                            if len(result.url.split("/")) > 2
                            else ""
                        )
                        # Strip www. prefix to save space
                        if domain.startswith("www."):
                            domain = domain[4:]

                    html += f"""
                        <li>
                            <a href="{result.url}" target="_blank">{result.title}</a> ({domain})
                        </li>
                    """

                html += """
                    </ul>
                """

            html += """
                </div>
            """

        # Add synthesized information if available
        if state.synthesized_info:
            html += f"""
                <div class="section">
                    <h2>Synthesized Information</h2>
            """

            for question, info in state.synthesized_info.items():
                html += f"""
                    <h3>{question} <span class="confidence {info.confidence_level}">{info.confidence_level}</span></h3>
                    <div class="synthesized">
                        <p>{info.synthesized_answer}</p>
                """

                if info.key_sources:
                    html += """
                        <div class="sources">
                            <p><strong>Key Sources:</strong></p>
                            <ul>
                    """

                    for source in info.key_sources[:3]:
                        html += f"""
                                <li><a href="{source}" target="_blank">{source[:50]}...</a></li>
                        """

                    if len(info.key_sources) > 3:
                        html += f"<li><em>...and {len(info.key_sources) - 3} more sources</em></li>"

                    html += """
                            </ul>
                        </div>
                    """

                if info.information_gaps:
                    html += f"""
                        <div class="metadata">
                            <p><strong>Information Gaps:</strong> {info.information_gaps}</p>
                        </div>
                    """

                html += """
                    </div>
                """

            html += """
                </div>
            """

        # Add viewpoint analysis if available
        if state.viewpoint_analysis:
            html += """
                <div class="section">
                    <h2>Viewpoint Analysis</h2>
                    <div class="viewpoints">
            """

            # Points of agreement
            if state.viewpoint_analysis.main_points_of_agreement:
                html += """
                        <h3>Points of Agreement</h3>
                        <ul>
                """

                for point in state.viewpoint_analysis.main_points_of_agreement:
                    html += f"""
                            <li>{point}</li>
                    """

                html += """
                        </ul>
                """

            # Areas of tension
            if state.viewpoint_analysis.areas_of_tension:
                html += """
                        <h3>Areas of Tension</h3>
                """

                for tension in state.viewpoint_analysis.areas_of_tension:
                    html += f"""
                        <div class="sub-section">
                            <h4>{tension.topic}</h4>
                            <ul>
                    """

                    for viewpoint, description in tension.viewpoints.items():
                        html += f"""
                                <li><strong>{viewpoint}:</strong> {description}</li>
                        """

                    html += """
                            </ul>
                        </div>
                    """

            # Perspective gaps and integrative insights
            if state.viewpoint_analysis.perspective_gaps:
                html += f"""
                        <h3>Perspective Gaps</h3>
                        <p>{state.viewpoint_analysis.perspective_gaps}</p>
                """

            if state.viewpoint_analysis.integrative_insights:
                html += f"""
                        <h3>Integrative Insights</h3>
                        <p>{state.viewpoint_analysis.integrative_insights}</p>
                """

            html += """
                    </div>
                </div>
            """

        # Add reflection results if available
        if state.enhanced_info and state.reflection_metadata:
            html += """
                <div class="section">
                    <h2>Reflection & Enhancement</h2>
            """

            # Reflection metadata
            if state.reflection_metadata:
                html += """
                    <div class="reflection">
                """

                if state.reflection_metadata.critique_summary:
                    html += """
                        <h3>Critique Summary</h3>
                        <ul>
                    """

                    for critique in state.reflection_metadata.critique_summary:
                        html += f"""
                            <li>{critique}</li>
                        """

                    html += """
                        </ul>
                    """

                if state.reflection_metadata.additional_questions_identified:
                    html += """
                        <h3>Additional Questions Identified</h3>
                        <ul>
                    """

                    for question in state.reflection_metadata.additional_questions_identified:
                        html += f"""
                            <li>{question}</li>
                        """

                    html += """
                        </ul>
                    """

                html += f"""
                        <div class="metadata">
                            <p><strong>Searches Performed:</strong> {len(state.reflection_metadata.searches_performed)}</p>
                            <p><strong>Improvements Made:</strong> {state.reflection_metadata.improvements_made}</p>
                        </div>
                    </div>
                """

            # Enhanced information
            if state.enhanced_info:
                html += """
                    <h3>Enhanced Information</h3>
                """

                for question, info in state.enhanced_info.items():
                    # Show only for questions with improvements
                    if info.improvements:
                        html += f"""
                        <div class="synthesized">
                            <h4>{question} <span class="confidence {info.confidence_level}">{info.confidence_level}</span></h4>
                            
                            <div class="improvements">
                                <p><strong>Improvements Made:</strong></p>
                                <ul>
                        """

                        for improvement in info.improvements:
                            html += f"""
                                    <li>{improvement}</li>
                            """

                        html += """
                                </ul>
                            </div>
                        </div>
                        """

            html += """
                </div>
            """

        # Add final report section if available
        if state.final_report_html:
            html += """
                <div class="section">
                    <h2>Final Report</h2>
                    <p><em>Final HTML report is available but not displayed here. View the HTML artifact to see the complete report.</em></p>
                </div>
            """

        # Close HTML tags
        html += """
            </div>
        </body>
        </html>
        """

        return html

    def _get_stage_class(self, state: ResearchState, stage: str) -> str:
        """Get CSS class for a stage based on current progress.

        Args:
            state: ResearchState object
            stage: Stage name

        Returns:
            CSS class string
        """
        current_stage = state.get_current_stage()

        # These are the stages in order
        stages = [
            "empty",
            "initial",
            "after_query_decomposition",
            "after_search",
            "after_synthesis",
            "after_viewpoint_analysis",
            "after_reflection",
            "final_report",
        ]

        current_index = (
            stages.index(current_stage) if current_stage in stages else 0
        )
        stage_index = stages.index(stage) if stage in stages else 0

        if stage_index == current_index:
            return "active"
        elif stage_index < current_index:
            return "completed"
        else:
            return ""

    def _calculate_progress(self, state: ResearchState) -> int:
        """Calculate overall progress percentage.

        Args:
            state: ResearchState object

        Returns:
            Progress percentage (0-100)
        """
        # Map stages to progress percentages
        stage_percentages = {
            "empty": 0,
            "initial": 5,
            "after_query_decomposition": 20,
            "after_search": 40,
            "after_synthesis": 60,
            "after_viewpoint_analysis": 75,
            "after_reflection": 90,
            "final_report": 100,
        }

        current_stage = state.get_current_stage()
        return stage_percentages.get(current_stage, 0)
