import os
import json
from typing import Dict, Any, Type, List, Union, Tuple, Optional

from zenml.enums import VisualizationType, ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer
from utils.data_models import Search


class ResearchMaterializer(BaseMaterializer):
    """Materializer for handling various research data types with visualizations."""

    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save(self, data: Any) -> None:
        """Save any research data to storage.

        Args:
            data: The data to save
        """
        # Determine the data type and save accordingly
        data_path = os.path.join(self.uri, "data.json")

        # Handle different data types
        if isinstance(data, list):
            # Convert Search objects to dicts if present
            processed_data = self._process_data_for_json(data)

            with self.artifact_store.open(data_path, "w") as f:
                json.dump(processed_data, f, indent=2)

        elif isinstance(data, dict):
            # Process dictionary contents (may contain Search objects)
            processed_data = self._process_data_for_json(data)

            with self.artifact_store.open(data_path, "w") as f:
                json.dump(processed_data, f, indent=2)

        elif isinstance(data, tuple):
            # Save each tuple item separately
            for i, item in enumerate(data):
                item_path = os.path.join(self.uri, f"item_{i}.json")
                processed_item = self._process_data_for_json(item)

                with self.artifact_store.open(item_path, "w") as f:
                    json.dump(processed_item, f, indent=2)

            # Save tuple structure
            structure_path = os.path.join(self.uri, "structure.json")
            with self.artifact_store.open(structure_path, "w") as f:
                json.dump({"type": "tuple", "length": len(data)}, f, indent=2)
        else:
            # Try to convert to JSON serializable format
            try:
                processed_data = self._process_data_for_json(data)
                with self.artifact_store.open(data_path, "w") as f:
                    json.dump(processed_data, f, indent=2)
            except:
                # Fallback to string representation
                with self.artifact_store.open(data_path, "w") as f:
                    f.write(str(data))

        # Create and save visualizations
        self._save_visualizations(data)

    def load(self, data_type: Type[Any]) -> Any:
        """Load research data from storage.

        Args:
            data_type: The type of data to load

        Returns:
            The loaded data
        """
        # Check if this is a tuple
        structure_path = os.path.join(self.uri, "structure.json")
        if fileio.exists(structure_path):
            with self.artifact_store.open(structure_path, "r") as f:
                structure = json.load(f)

            if structure.get("type") == "tuple":
                # Load tuple items
                items = []
                for i in range(structure.get("length", 0)):
                    item_path = os.path.join(self.uri, f"item_{i}.json")
                    if fileio.exists(item_path):
                        with self.artifact_store.open(item_path, "r") as f:
                            items.append(json.load(f))

                return tuple(items)

        # Standard data path
        data_path = os.path.join(self.uri, "data.json")

        if fileio.exists(data_path):
            try:
                with self.artifact_store.open(data_path, "r") as f:
                    return json.load(f)
            except:
                # Fallback to string
                with self.artifact_store.open(data_path, "r") as f:
                    return f.read()

        # Return empty data if nothing found
        if data_type == list:
            return []
        elif data_type == dict:
            return {}
        elif data_type == tuple:
            return tuple()
        else:
            return None

    def _process_data_for_json(self, data: Any) -> Any:
        """Process data to make it JSON serializable.

        Args:
            data: The data to process

        Returns:
            JSON serializable version of the data
        """
        if isinstance(data, list):
            return [self._process_data_for_json(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._process_data_for_json(v) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple(self._process_data_for_json(item) for item in data)
        elif isinstance(data, Search):
            return {"url": data.url, "content": data.content}
        elif hasattr(data, "__dict__"):
            return self._process_data_for_json(data.__dict__)
        else:
            return data

    def _save_visualizations(self, data: Any) -> Dict[str, VisualizationType]:
        """Create and save visualizations for research data.

        Args:
            data: The data to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualizations = {}

        # Create HTML visualization based on data type
        html_path = os.path.join(self.uri, "visualization.html")

        if isinstance(data, list):
            html_content = self._generate_list_visualization(data)
        elif isinstance(data, dict):
            html_content = self._generate_dict_visualization(data)
        elif isinstance(data, tuple):
            # Extract tuple items for visualization
            if len(data) == 2:
                html_content = self._generate_tuple_visualization(data)
            else:
                html_content = self._generate_basic_visualization(data)
        else:
            html_content = self._generate_basic_visualization(data)

        # Write the HTML content to a file
        with fileio.open(html_path, "w") as f:
            f.write(html_content)

        visualizations[html_path] = VisualizationType.HTML
        return visualizations

    def _generate_list_visualization(self, data: List) -> str:
        """Generate HTML visualization for a list.

        Args:
            data: The list data to visualize

        Returns:
            HTML string
        """
        title = "Research Data: List"

        # Generate list items HTML
        items_html = ""
        for i, item in enumerate(data):
            if isinstance(item, dict):
                item_html = f"<li><strong>Item {i + 1}:</strong> {self._generate_dict_content(item)}</li>"
            elif isinstance(item, str):
                item_html = f"<li>{item}</li>"
            else:
                item_html = f"<li>{str(item)}</li>"
            items_html += item_html

        return self._html_template(
            title,
            f"""
        <div class="data-container">
            <h2>List Items ({len(data)})</h2>
            <ul>
                {items_html}
            </ul>
        </div>
        """,
        )

    def _generate_dict_visualization(self, data: Dict) -> str:
        """Generate HTML visualization for a dictionary.

        Args:
            data: The dictionary data to visualize

        Returns:
            HTML string
        """
        title = "Research Data: Dictionary"
        content = self._generate_dict_content(data)

        return self._html_template(
            title,
            f"""
        <div class="data-container">
            <h2>Dictionary Content ({len(data)} keys)</h2>
            {content}
        </div>
        """,
        )

    def _generate_dict_content(self, data: Dict, depth: int = 0) -> str:
        """Generate HTML content for a dictionary.

        Args:
            data: The dictionary to convert to HTML
            depth: Current nesting depth

        Returns:
            HTML string
        """
        # Don't go too deep in nested structures
        if depth > 3:
            return (
                f"<div class='nested'>Nested data ({len(data)} items)...</div>"
            )

        html = "<div class='dict-content'>"
        for key, value in data.items():
            if isinstance(value, dict):
                nested_content = self._generate_dict_content(value, depth + 1)
                html += f"<div class='dict-item'><strong>{key}:</strong> {nested_content}</div>"
            elif isinstance(value, list):
                list_content = f"<ul class='nested-list'>"
                for item in value[:5]:  # Show first 5 only
                    list_content += f"<li>{str(item)[:100]}</li>"
                if len(value) > 5:
                    list_content += (
                        f"<li>... and {len(value) - 5} more items</li>"
                    )
                list_content += "</ul>"
                html += f"<div class='dict-item'><strong>{key}:</strong> {list_content}</div>"
            else:
                # Truncate very long values
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:197] + "..."
                html += f"<div class='dict-item'><strong>{key}:</strong> {value_str}</div>"
        html += "</div>"
        return html

    def _generate_tuple_visualization(self, data: Tuple) -> str:
        """Generate HTML visualization for a tuple.

        Args:
            data: The tuple data to visualize

        Returns:
            HTML string
        """
        title = "Research Data: Enhanced Synthesis Results"

        # Assuming we're visualizing the output of iterative_reflection_step
        if len(data) == 2:
            enhanced_info, reflection_metadata = data

            # Generate content for enhanced_info (first element)
            info_html = ""
            for question, details in enhanced_info.items():
                info_html += f"<div class='question-section'>"
                info_html += f"<h3>{question}</h3>"

                if isinstance(details, dict):
                    confidence = details.get("confidence_level", "unknown")
                    answer = details.get(
                        "synthesized_answer", "No answer available"
                    )
                    sources = details.get("key_sources", [])

                    info_html += f"<div class='answer'><p>{answer}</p></div>"
                    info_html += f"<div class='metadata'>"
                    info_html += f"<p><strong>Confidence:</strong> <span class='confidence {confidence}'>{confidence}</span></p>"

                    # Add improvements if available
                    improvements = details.get("improvements", [])
                    if improvements:
                        info_html += f"<p><strong>Improvements ({len(improvements)}):</strong></p>"
                        info_html += "<ul class='improvements'>"
                        for improvement in improvements:
                            info_html += f"<li>{improvement}</li>"
                        info_html += "</ul>"

                    # Add sources if available
                    if sources:
                        info_html += f"<p><strong>Sources ({len(sources)}):</strong></p>"
                        info_html += "<ul class='sources'>"
                        for i, source in enumerate(sources[:3]):
                            info_html += f"<li><a href='{source}' target='_blank'>{source[:50]}...</a></li>"
                        if len(sources) > 3:
                            info_html += f"<li>... and {len(sources) - 3} more sources</li>"
                        info_html += "</ul>"

                    info_html += "</div>"
                else:
                    info_html += f"<p>{str(details)}</p>"

                info_html += "</div>"

            # Generate content for reflection_metadata (second element)
            meta_html = ""
            if isinstance(reflection_metadata, dict):
                # Handle critique summary
                critiques = reflection_metadata.get("critique_summary", [])
                if critiques:
                    meta_html += "<div class='critique-section'>"
                    meta_html += (
                        f"<h3>Critique Summary ({len(critiques)})</h3>"
                    )
                    meta_html += "<ul class='critique-list'>"
                    for critique in critiques:
                        meta_html += f"<li>{critique}</li>"
                    meta_html += "</ul>"
                    meta_html += "</div>"

                # Handle additional questions
                questions = reflection_metadata.get(
                    "additional_questions_identified", []
                )
                if questions:
                    meta_html += "<div class='additional-questions'>"
                    meta_html += f"<h3>Additional Questions Identified ({len(questions)})</h3>"
                    meta_html += "<ul class='question-list'>"
                    for question in questions:
                        meta_html += f"<li>{question}</li>"
                    meta_html += "</ul>"
                    meta_html += "</div>"

                # Show other metadata
                searches = reflection_metadata.get("searches_performed", [])
                improvements = reflection_metadata.get("improvements_made", 0)

                meta_html += "<div class='stats'>"
                meta_html += f"<p><strong>Additional Searches:</strong> {len(searches)}</p>"
                meta_html += f"<p><strong>Total Improvements:</strong> {improvements}</p>"
                meta_html += "</div>"

            return self._html_template(
                title,
                f"""
            <div class="summary-container">
                <h2>Reflection Results</h2>
                <div class="reflection-metadata">
                    {meta_html}
                </div>
                
                <h2>Enhanced Synthesized Information</h2>
                <div class="synthesized-info">
                    {info_html}
                </div>
            </div>
            """,
            )

        return self._generate_basic_visualization(data)

    def _generate_basic_visualization(self, data: Any) -> str:
        """Generate basic HTML visualization for any data type.

        Args:
            data: The data to visualize

        Returns:
            HTML string
        """
        title = "Research Data"
        content = f"<pre>{str(data)[:2000]}</pre>"
        if len(str(data)) > 2000:
            content += "<p><em>Content truncated due to size...</em></p>"

        return self._html_template(
            title,
            f"""
        <div class="data-container">
            <h2>Data Content</h2>
            <div class="data-preview">
                {content}
            </div>
        </div>
        """,
        )

    def _html_template(self, title: str, content: str) -> str:
        """Generate HTML template with standard styling.

        Args:
            title: Page title
            content: Main content

        Returns:
            Complete HTML document as string
        """
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
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
                    margin-top: 20px;
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .data-container, .summary-container {{
                    margin-top: 20px;
                }}
                .dict-content {{
                    margin-left: 20px;
                }}
                .dict-item {{
                    margin: 8px 0;
                }}
                .nested {{
                    color: #7f8c8d;
                    font-style: italic;
                }}
                .nested-list {{
                    margin: 5px 0 5px 20px;
                }}
                pre {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-size: 14px;
                }}
                .question-section {{
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .answer {{
                    margin: 10px 0;
                }}
                .metadata {{
                    font-size: 14px;
                    color: #555;
                    margin-top: 10px;
                    padding-top: 10px;
                    border-top: 1px solid #eee;
                }}
                .confidence {{
                    padding: 2px 6px;
                    border-radius: 3px;
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
                .critique-section, .additional-questions, .stats {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #eef2f7;
                    border-radius: 8px;
                }}
                .improvements, .sources, .critique-list, .question-list {{
                    margin-top: 5px;
                }}
                a {{
                    color: #3498db;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                {content}
            </div>
        </body>
        </html>
        """
