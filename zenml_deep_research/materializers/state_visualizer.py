import os
import json
from dataclasses import asdict
from typing import Dict, Any, Type

from zenml.enums import VisualizationType, ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from utils.data_models import State, Paragraph, Research, Search


class StateMaterializer(BaseMaterializer):
    """Materializer that handles State objects with visualizations."""

    ASSOCIATED_TYPES = (State,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save(self, data: State) -> None:
        """Save the State data to storage.

        Args:
            data: The State object to save
        """
        # Convert State dataclass to dictionary using dataclasses.asdict
        state_dict = asdict(data)

        # Save the state dictionary as JSON
        state_path = os.path.join(self.uri, "state.json")
        with self.artifact_store.open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        # Save visualizations
        self.save_visualizations(data)

    def load(self, data_type: Type[Any]) -> State:
        """Load the State data from storage.

        Args:
            data_type: The type of data to load

        Returns:
            The loaded State object
        """
        # Path to the saved state
        state_path = os.path.join(self.uri, "state.json")

        # Load the serialized state
        with self.artifact_store.open(state_path, "r") as f:
            state_dict = json.load(f)

        # Reconstruct the State object
        paragraphs = []
        for p_dict in state_dict["paragraphs"]:
            # Convert the nested research dictionary
            research_dict = p_dict["research"]
            research_dict["search_history"] = [
                Search(**s) for s in research_dict["search_history"]
            ]

            # Create paragraph with research
            paragraphs.append(
                Paragraph(
                    title=p_dict["title"],
                    content=p_dict["content"],
                    research=Research(**research_dict),
                )
            )

        # Return the complete State object
        return State(
            report_title=state_dict["report_title"],
            query=state_dict["query"],
            paragraphs=paragraphs,
        )

    def save_visualizations(self, data: State) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the State object.

        Args:
            data: The State object to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization of the report structure
        visualization_path = os.path.join(self.uri, "report_structure.html")

        # Create HTML content with styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Report Structure: {data.report_title}</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
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
                .query {{
                    background-color: #eef2f7;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                .paragraph {{
                    margin-bottom: 20px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 15px;
                }}
                .paragraph:last-child {{
                    border-bottom: none;
                }}
                .paragraph h3 {{
                    margin-bottom: 10px;
                }}
                .content {{
                    color: #555;
                    margin-left: 10px;
                }}
                .meta {{
                    font-size: 12px;
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{data.report_title or "Research Report"}</h1>
                <div class="query">
                    <h2>Research Query:</h2>
                    <p>{data.query}</p>
                </div>
                
                <h2>Report Structure:</h2>
        """

        # Add each paragraph to the HTML
        for i, paragraph in enumerate(data.paragraphs):
            html_content += f"""
                <div class="paragraph">
                    <h3>{i + 1}. {paragraph.title}</h3>
                    <div class="content">{paragraph.content}</div>
            """

            # Add research information if available
            if paragraph.research.latest_summary:
                html_content += f"""
                    <div class="meta">
                        <p><strong>Latest Summary:</strong> {paragraph.research.latest_summary[:100]}{"..." if len(paragraph.research.latest_summary) > 100 else ""}</p>
                        <p><strong>Reflection Iterations:</strong> {paragraph.research.reflection_iteration}</p>
                        <p><strong>Sources:</strong> {len(paragraph.research.search_history)}</p>
                    </div>
                """

            html_content += """
                </div>
            """

        # Close HTML tags
        html_content += """
            </div>
        </body>
        </html>
        """

        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}
