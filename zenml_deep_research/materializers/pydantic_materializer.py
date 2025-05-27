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
                html, body {{
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    min-height: 100%;
                }}
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    background-color: #f5f5f5;
                    color: #333;
                    display: flex;
                    flex-direction: column;
                }}
                .container {{
                    background-color: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    max-width: 1400px;
                    margin: 20px auto;
                    width: calc(100% - 40px);
                    flex: 1;
                    display: flex;
                    flex-direction: column;
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
                .stage.completed {{
                    background-color: #3498db;
                    color: white;
                }}
                
                /* Tab styles */
                .tabs {{
                    display: flex;
                    list-style: none;
                    padding: 0;
                    margin: 20px 0 0 0;
                    border-bottom: 2px solid #ddd;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    background-color: #f0f0f0;
                    border: 1px solid #ddd;
                    border-bottom: none;
                    margin-right: 5px;
                    border-radius: 5px 5px 0 0;
                    transition: background-color 0.3s;
                }}
                .tab:hover {{
                    background-color: #e0e0e0;
                }}
                .tab.active {{
                    background-color: white;
                    border-bottom: 2px solid white;
                    margin-bottom: -2px;
                    font-weight: bold;
                }}
                .tab-content {{
                    display: none;
                    padding: 20px 0;
                    flex: 1;
                    overflow-y: auto;
                    min-height: 500px;
                }}
                .tab-content.active {{
                    display: flex;
                    flex-direction: column;
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
                .sub-section {{
                    margin: 15px 0;
                }}
                .improvements {{
                    margin-top: 10px;
                }}
            </style>
            <script>
                function showTab(tabName) {{
                    // Hide all tab contents
                    var tabContents = document.getElementsByClassName('tab-content');
                    for (var i = 0; i < tabContents.length; i++) {{
                        tabContents[i].classList.remove('active');
                    }}
                    
                    // Remove active class from all tabs
                    var tabs = document.getElementsByClassName('tab');
                    for (var i = 0; i < tabs.length; i++) {{
                        tabs[i].classList.remove('active');
                    }}
                    
                    // Show the selected tab content
                    document.getElementById(tabName).classList.add('active');
                    
                    // Add active class to the clicked tab
                    document.getElementById(tabName + '-tab').classList.add('active');
                }}
            </script>
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
                
                <!-- Tab navigation -->
                <ul class="tabs">
        """

        # Determine which tab should be active based on current stage
        current_stage = state.get_current_stage()

        # Map stages to tabs
        stage_to_tab = {
            "empty": "overview",
            "initial": "overview",
            "after_query_decomposition": "sub-questions",
            "after_search": "search-results",
            "after_synthesis": "synthesis",
            "after_viewpoint_analysis": "viewpoints",
            "after_reflection": "reflection",
            "final_report": "final-report",
        }

        # Get the default active tab based on stage
        default_active_tab = stage_to_tab.get(current_stage, "overview")

        # Create tab headers dynamically based on available data
        tabs_created = []

        # Overview tab is always shown
        is_active = default_active_tab == "overview"
        html += f'<li class="tab {"active" if is_active else ""}" id="overview-tab" onclick="showTab(\'overview\')">Overview</li>'
        tabs_created.append("overview")

        if state.sub_questions:
            is_active = default_active_tab == "sub-questions"
            html += f'<li class="tab {"active" if is_active else ""}" id="sub-questions-tab" onclick="showTab(\'sub-questions\')">Sub-Questions</li>'
            tabs_created.append("sub-questions")

        if state.search_results:
            is_active = default_active_tab == "search-results"
            html += f'<li class="tab {"active" if is_active else ""}" id="search-results-tab" onclick="showTab(\'search-results\')">Search Results</li>'
            tabs_created.append("search-results")

        if state.synthesized_info:
            is_active = default_active_tab == "synthesis"
            html += f'<li class="tab {"active" if is_active else ""}" id="synthesis-tab" onclick="showTab(\'synthesis\')">Synthesis</li>'
            tabs_created.append("synthesis")

        if state.viewpoint_analysis:
            is_active = default_active_tab == "viewpoints"
            html += f'<li class="tab {"active" if is_active else ""}" id="viewpoints-tab" onclick="showTab(\'viewpoints\')">Viewpoints</li>'
            tabs_created.append("viewpoints")

        if state.enhanced_info or state.reflection_metadata:
            is_active = default_active_tab == "reflection"
            html += f'<li class="tab {"active" if is_active else ""}" id="reflection-tab" onclick="showTab(\'reflection\')">Reflection</li>'
            tabs_created.append("reflection")

        if state.final_report_html:
            is_active = default_active_tab == "final-report"
            html += f'<li class="tab {"active" if is_active else ""}" id="final-report-tab" onclick="showTab(\'final-report\')">Final Report</li>'
            tabs_created.append("final-report")

        # Ensure the active tab actually exists in the created tabs
        # If not, fallback to the first available tab
        if default_active_tab not in tabs_created and tabs_created:
            default_active_tab = tabs_created[0]

        html += """
                </ul>
                
                <!-- Tab content containers -->
        """

        # Overview tab content (always shown)
        is_active = default_active_tab == "overview"
        html += f"""
                <div id="overview" class="tab-content {"active" if is_active else ""}">
                    <div class="section">
                        <h2>Main Query</h2>
                        <div class="query">
        """

        if state.main_query:
            html += f"<p>{state.main_query}</p>"
        else:
            html += "<p><em>No main query specified</em></p>"

        html += """
                        </div>
                    </div>
                </div>
        """

        # Sub-questions tab content
        if state.sub_questions:
            is_active = default_active_tab == "sub-questions"
            html += f"""
                <div id="sub-questions" class="tab-content {"active" if is_active else ""}">
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
                </div>
            """

        # Search results tab content
        if state.search_results:
            is_active = default_active_tab == "search-results"
            html += f"""
                <div id="search-results" class="tab-content {"active" if is_active else ""}">
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
                    # Extract domain from URL or use special handling for generated content
                    if result.url == "tavily-generated-answer":
                        domain = "Tavily"
                    else:
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
                </div>
            """

        # Synthesized information tab content
        if state.synthesized_info:
            is_active = default_active_tab == "synthesis"
            html += f"""
                <div id="synthesis" class="tab-content {"active" if is_active else ""}">
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
                </div>
            """

        # Viewpoint analysis tab content
        if state.viewpoint_analysis:
            is_active = default_active_tab == "viewpoints"
            html += f"""
                <div id="viewpoints" class="tab-content {"active" if is_active else ""}">
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
                </div>
            """

        # Reflection & Enhancement tab content
        if state.enhanced_info or state.reflection_metadata:
            is_active = default_active_tab == "reflection"
            html += f"""
                <div id="reflection" class="tab-content {"active" if is_active else ""}">
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
                </div>
            """

        # Final report tab
        if state.final_report_html:
            is_active = default_active_tab == "final-report"
            html += f"""
                <div id="final-report" class="tab-content {"active" if is_active else ""}">
                    <div class="section">
                        <h2>Final Report</h2>
                        <p><em>Final HTML report is available but not displayed here. View the HTML artifact to see the complete report.</em></p>
                    </div>
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
