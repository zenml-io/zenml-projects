"""Materializer for MCPResult with HTML and JSON visualization."""

import json
import os
from typing import Dict

from utils.css_utils import get_shared_css_tag
from utils.pydantic_models import MCPResult
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class MCPResultMaterializer(PydanticMaterializer):
    """Materializer for MCPResult with interactive visualization."""

    ASSOCIATED_TYPES = (MCPResult,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: MCPResult
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the MCPResult.

        Args:
            data: The MCPResult to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualization_path = os.path.join(self.uri, "mcp_result.html")
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, data: MCPResult) -> str:
        """Generate HTML visualization for the MCP result.

        Args:
            data: The MCPResult to visualize

        Returns:
            HTML string
        """
        # Process the raw MCP result for pretty display
        raw_result_html = ""
        if data.raw_mcp_result:
            try:
                # Try to parse as JSON for pretty formatting
                parsed_json = json.loads(data.raw_mcp_result)
                formatted_json = json.dumps(
                    parsed_json, indent=2, ensure_ascii=False
                )
                raw_result_html = f"""
                <div class="dr-card">
                    <h2>Raw MCP Search Results</h2>
                    <div class="dr-section dr-section--info">
                        <h3>JSON Data</h3>
                        <pre class="json-display">{self._escape_html(formatted_json)}</pre>
                    </div>
                </div>
                """
            except json.JSONDecodeError:
                # If not valid JSON, display as plain text
                raw_result_html = f"""
                <div class="dr-card">
                    <h2>Raw MCP Search Results</h2>
                    <div class="dr-section dr-section--info">
                        <h3>Raw Data</h3>
                        <pre class="raw-display">{self._escape_html(data.raw_mcp_result)}</pre>
                    </div>
                </div>
                """
        else:
            raw_result_html = """
            <div class="dr-card">
                <h2>Raw MCP Search Results</h2>
                <div class="dr-empty">No raw search results available</div>
            </div>
            """

        # Display the MCP result (HTML formatted)
        mcp_result_html = ""
        if data.mcp_result:
            mcp_result_html = f"""
            <div class="dr-card">
                <h2>MCP Analysis Summary</h2>
                <div class="mcp-content">
                    {data.mcp_result}
                </div>
            </div>
            """
        else:
            mcp_result_html = """
            <div class="dr-card">
                <h2>MCP Analysis Summary</h2>
                <div class="dr-empty">No analysis summary available</div>
            </div>
            """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MCP Search Results</title>
            {get_shared_css_tag()}
            <style>
                /* Component-specific styles */
                body {{
                    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
                    min-height: 100vh;
                }}
                
                .mcp-content {{
                    background: var(--color-bg-secondary);
                    border-radius: var(--radius-md);
                    padding: var(--spacing-lg);
                    margin-top: var(--spacing-md);
                    line-height: 1.8;
                }}
                
                .mcp-content p {{
                    margin: 1em 0;
                    color: var(--color-text-primary);
                }}
                
                .mcp-content p:first-child {{
                    margin-top: 0;
                }}
                
                .mcp-content p:last-child {{
                    margin-bottom: 0;
                }}
                
                .json-display, .raw-display {{
                    background: #282c34;
                    color: #abb2bf;
                    padding: var(--spacing-md);
                    border-radius: var(--radius-md);
                    overflow-x: auto;
                    font-size: 0.875rem;
                    line-height: 1.5;
                    max-height: 600px;
                    overflow-y: auto;
                    font-family: 'Courier New', Courier, monospace;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }}
                
                .json-display {{
                    border: 1px solid var(--color-border);
                }}
                
                /* Syntax highlighting for JSON */
                .json-key {{
                    color: #e06c75;
                }}
                
                .json-string {{
                    color: #98c379;
                }}
                
                .json-number {{
                    color: #d19a66;
                }}
                
                .json-boolean {{
                    color: #56b6c2;
                }}
                
                .json-null {{
                    color: #c678dd;
                }}
                
                .dr-card + .dr-card {{
                    margin-top: var(--spacing-lg);
                }}
                
                /* Header styling */
                .mcp-header {{
                    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
                    color: white;
                    padding: var(--spacing-lg);
                    border-radius: var(--radius-lg);
                    text-align: center;
                    margin-bottom: var(--spacing-xl);
                    box-shadow: var(--shadow-lg);
                }}
                
                .mcp-header h1 {{
                    margin: 0;
                    font-size: 2rem;
                }}
                
                .mcp-header p {{
                    margin: 0.5rem 0 0 0;
                    opacity: 0.9;
                }}
            </style>
        </head>
        <body>
            <div class="dr-container">
                <div class="mcp-header">
                    <h1>MCP Search Results</h1>
                    <p>Additional research performed using MCP tools</p>
                </div>
                
                {mcp_result_html}
                
                {raw_result_html}
            </div>
        </body>
        </html>
        """

        return html

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters in text.

        Args:
            text: Text to escape

        Returns:
            Escaped text safe for HTML display
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
