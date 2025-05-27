"""Materializer for ReflectionOutput with custom visualization."""

import os
from typing import Dict

from utils.pydantic_models import ReflectionOutput
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class ReflectionOutputMaterializer(PydanticMaterializer):
    """Materializer for the ReflectionOutput class with visualizations."""

    ASSOCIATED_TYPES = (ReflectionOutput,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: ReflectionOutput
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the ReflectionOutput.

        Args:
            data: The ReflectionOutput to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization
        visualization_path = os.path.join(self.uri, "reflection_output.html")

        # Create HTML content
        html_content = self._generate_visualization_html(data)

        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, output: ReflectionOutput) -> str:
        """Generate HTML visualization for the reflection output.

        Args:
            output: The ReflectionOutput to visualize

        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reflection Output</title>
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
                    border-bottom: 2px solid #e74c3c;
                    padding-bottom: 10px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                }}
                .recommended-queries {{
                    border-left: 4px solid #27ae60;
                }}
                .critique-summary {{
                    border-left: 4px solid #e74c3c;
                }}
                .additional-questions {{
                    border-left: 4px solid #3498db;
                }}
                .query-item {{
                    background-color: #e8f8f5;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                }}
                .query-number {{
                    background-color: #27ae60;
                    color: white;
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 15px;
                    font-weight: bold;
                }}
                .critique-item {{
                    background-color: #fdedec;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 4px;
                }}
                .critique-label {{
                    font-weight: bold;
                    color: #c0392b;
                    margin-bottom: 5px;
                }}
                .question-item {{
                    background-color: #ebf5fb;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                    border-left: 3px solid #3498db;
                }}
                .metadata {{
                    font-size: 12px;
                    color: #7f8c8d;
                    margin-top: 20px;
                    padding-top: 10px;
                    border-top: 1px dashed #ddd;
                }}
                .state-info {{
                    background-color: #f0f0f0;
                    padding: 10px;
                    border-radius: 4px;
                    margin-top: 10px;
                }}
                .count-badge {{
                    display: inline-block;
                    background-color: #95a5a6;
                    color: white;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    margin-left: 10px;
                }}
                pre {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 14px;
                }}
                .icon {{
                    margin-right: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Reflection & Analysis Output</h1>
                
                <div class="section recommended-queries">
                    <h2>
                        <span class="icon">✅</span>Recommended Additional Queries
                        <span class="count-badge">{len(output.recommended_queries)}</span>
                    </h2>
        """

        if output.recommended_queries:
            for i, query in enumerate(output.recommended_queries, 1):
                html += f"""
                    <div class="query-item">
                        <div class="query-number">{i}</div>
                        <div>{query}</div>
                    </div>
                """
        else:
            html += """
                    <p style="color: #7f8c8d; font-style: italic;">No additional queries recommended</p>
            """

        html += """
                </div>
                
                <div class="section critique-summary">
                    <h2>
                        <span class="icon">📝</span>Critique Summary
                        <span class="count-badge">{}</span>
                    </h2>
        """.format(len(output.critique_summary))

        if output.critique_summary:
            for critique in output.critique_summary:
                html += """
                    <div class="critique-item">
                """

                # Handle different critique formats
                if isinstance(critique, dict):
                    for key, value in critique.items():
                        html += f"""
                        <div class="critique-label">{key}:</div>
                        <div>{value}</div>
                        """
                else:
                    html += f"""
                        <div>{critique}</div>
                    """

                html += """
                    </div>
                """
        else:
            html += """
                    <p style="color: #7f8c8d; font-style: italic;">No critique summary available</p>
            """

        html += """
                </div>
                
                <div class="section additional-questions">
                    <h2>
                        <span class="icon">❓</span>Additional Questions Identified
                        <span class="count-badge">{}</span>
                    </h2>
        """.format(len(output.additional_questions))

        if output.additional_questions:
            for question in output.additional_questions:
                html += f"""
                    <div class="question-item">
                        {question}
                    </div>
                """
        else:
            html += """
                    <p style="color: #7f8c8d; font-style: italic;">No additional questions identified</p>
            """

        html += """
                </div>
                
                <div class="section">
                    <h2><span class="icon">📊</span>Research State Summary</h2>
                    <div class="state-info">
                        <p><strong>Main Query:</strong> {}</p>
                        <p><strong>Current Stage:</strong> {}</p>
                        <p><strong>Sub-questions:</strong> {}</p>
                        <p><strong>Search Results:</strong> {} queries with results</p>
                        <p><strong>Synthesized Info:</strong> {} topics synthesized</p>
                    </div>
                </div>
        """.format(
            output.state.main_query,
            output.state.get_current_stage().replace("_", " ").title(),
            len(output.state.sub_questions),
            len(output.state.search_results),
            len(output.state.synthesized_info),
        )

        # Add metadata section
        html += """
                <div class="metadata">
                    <p><em>This reflection output suggests improvements and additional research directions based on the current research state.</em></p>
                </div>
            </div>
        </body>
        </html>
        """

        return html
