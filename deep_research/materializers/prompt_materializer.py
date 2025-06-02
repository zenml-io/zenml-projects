"""Materializer for individual Prompt with custom HTML visualization.

This module provides a materializer that creates beautiful HTML visualizations
for individual prompts in the ZenML dashboard.
"""

import os
from typing import Dict

from utils.css_utils import (
    create_stat_card,
    get_card_class,
    get_grid_class,
    get_shared_css_tag,
)
from utils.pydantic_models import Prompt
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class PromptMaterializer(PydanticMaterializer):
    """Materializer for Prompt with custom visualization."""

    ASSOCIATED_TYPES = (Prompt,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: Prompt
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the Prompt.

        Args:
            data: The Prompt to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization
        visualization_path = os.path.join(self.uri, "prompt.html")

        # Create HTML content
        html_content = self._generate_visualization_html(data)

        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, prompt: Prompt) -> str:
        """Generate HTML visualization for a single prompt.

        Args:
            prompt: The Prompt to visualize

        Returns:
            HTML string
        """
        # Create tags HTML
        tag_html = ""
        if prompt.tags:
            tag_html = '<div class="prompt-tags">'
            for tag in prompt.tags:
                tag_html += (
                    f'<span class="dr-tag dr-tag--primary">{tag}</span>'
                )
            tag_html += "</div>"

        # Build stats HTML
        stats_html = f"""
        <div class="{get_grid_class("stats")}">
            {create_stat_card(len(prompt.content.split()), "Words")}
            {create_stat_card(len(prompt.content), "Characters")}
            {create_stat_card(len(prompt.content.splitlines()), "Lines")}
        </div>
        """

        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{prompt.name} - Prompt</title>
            {get_shared_css_tag()}
            <style>
                /* Component-specific styles */
                .prompt-header {{
                    border-bottom: 3px solid var(--color-border);
                    padding-bottom: var(--spacing-md);
                    margin-bottom: var(--spacing-lg);
                }}
                
                .prompt-title {{
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    flex-wrap: wrap;
                }}
                
                .prompt-version {{
                    background-color: #e3f2fd;
                    color: #1976d2;
                    padding: 6px 15px;
                    border-radius: 20px;
                    font-size: 0.5em;
                    font-weight: 500;
                }}
                
                .prompt-description {{
                    color: var(--color-text-secondary);
                    font-size: 1.1em;
                    margin-bottom: var(--spacing-md);
                    font-style: italic;
                }}
                
                .prompt-tags {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: var(--spacing-xs);
                    margin-bottom: var(--spacing-md);
                }}
                
                .prompt-content-section {{
                    margin-top: var(--spacing-lg);
                }}
                
                .prompt-content {{
                    background-color: var(--color-bg-secondary);
                    border: 1px solid var(--color-border);
                    border-radius: var(--radius-md);
                    padding: var(--spacing-md);
                    font-family: var(--font-family-mono);
                    font-size: 0.95em;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    position: relative;
                    max-height: 600px;
                    overflow-y: auto;
                }}
                
                .copy-button {{
                    position: absolute;
                    top: 15px;
                    right: 15px;
                }}
                
                .copy-button.copied {{
                    background-color: var(--color-success);
                }}
                
                .copy-icon {{
                    width: 16px;
                    height: 16px;
                }}
                
                .section-icon {{
                    margin-right: var(--spacing-xs);
                }}
                
                @media (max-width: 768px) {{
                    .prompt-title {{
                        flex-direction: column;
                        align-items: flex-start;
                        gap: 10px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="{get_card_class()} dr-container dr-container--narrow">
                <div class="prompt-header">
                    <h1 class="prompt-title">
                        🎯 {prompt.name}
                        <span class="prompt-version">v{prompt.version}</span>
                    </h1>
                    {f'<p class="prompt-description">{prompt.description}</p>' if prompt.description else ""}
                    {tag_html}
                </div>
                
                {stats_html}
                
                <div class="prompt-content-section">
                    <h2><span class="section-icon">📝</span>Prompt Content</h2>
                    <div class="prompt-content" id="promptContent">
                        <button class="dr-button dr-button--small copy-button" onclick="copyToClipboard()" id="copyButton">
                            <svg class="copy-icon" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z"></path>
                                <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z"></path>
                            </svg>
                            Copy
                        </button>
                        {self._escape_html(prompt.content)}
                    </div>
                </div>
            </div>
            
            <script>
                function copyToClipboard() {{
                    const content = document.getElementById('promptContent');
                    const button = document.getElementById('copyButton');
                    const text = content.textContent.replace('Copy', '').trim();
                    
                    navigator.clipboard.writeText(text).then(() => {{
                        button.innerHTML = `
                            <svg class="copy-icon" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd"></path>
                            </svg>
                            Copied!
                        `;
                        button.classList.add('copied');
                        
                        setTimeout(() => {{
                            button.innerHTML = `
                                <svg class="copy-icon" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z"></path>
                                    <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z"></path>
                                </svg>
                                Copy
                            `;
                            button.classList.remove('copied');
                        }}, 2000);
                    }}).catch(err => {{
                        console.error('Failed to copy:', err);
                        alert('Failed to copy to clipboard');
                    }});
                }}
            </script>
        </body>
        </html>
        """

        return html

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
