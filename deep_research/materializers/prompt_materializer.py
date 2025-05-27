"""Materializer for individual Prompt with custom HTML visualization.

This module provides a materializer that creates beautiful HTML visualizations
for individual prompts in the ZenML dashboard.
"""

import os
from typing import Dict

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
        # Determine tag colors
        tag_html = ""
        if prompt.tags:
            tag_colors = {
                "search": "search",
                "synthesis": "synthesis",
                "analysis": "analysis",
                "reflection": "reflection",
                "report": "report",
                "query": "query",
                "decomposition": "decomposition",
                "viewpoint": "viewpoint",
                "conclusion": "conclusion",
                "summary": "summary",
                "introduction": "introduction",
            }

            tag_html = '<div class="prompt-tags">'
            for tag in prompt.tags:
                tag_class = tag_colors.get(tag, "default")
                tag_html += f'<span class="tag {tag_class}">{tag}</span>'
            tag_html += "</div>"

        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{prompt.name} - Prompt</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f7fa;
                    color: #333;
                }}
                
                .container {{
                    background-color: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
                }}
                
                .prompt-header {{
                    border-bottom: 3px solid #e1e8ed;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                
                h1 {{
                    color: #2c3e50;
                    margin: 0 0 15px 0;
                    font-size: 2.2em;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
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
                    color: #6c757d;
                    font-size: 1.1em;
                    margin-bottom: 20px;
                    font-style: italic;
                }}
                
                .prompt-tags {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin-bottom: 20px;
                }}
                
                .tag {{
                    background-color: #f0f0f0;
                    color: #555;
                    padding: 5px 12px;
                    border-radius: 15px;
                    font-size: 0.9em;
                    font-weight: 500;
                }}
                
                .tag.default {{ background-color: #f0f0f0; color: #555; }}
                .tag.search {{ background-color: #fff3cd; color: #856404; }}
                .tag.synthesis {{ background-color: #d4edda; color: #155724; }}
                .tag.analysis {{ background-color: #d1ecf1; color: #0c5460; }}
                .tag.reflection {{ background-color: #f8d7da; color: #721c24; }}
                .tag.report {{ background-color: #e2e3e5; color: #383d41; }}
                .tag.query {{ background-color: #f3e5f5; color: #6a1b9a; }}
                .tag.decomposition {{ background-color: #e8f5e9; color: #2e7d32; }}
                .tag.viewpoint {{ background-color: #fff8e1; color: #f57c00; }}
                .tag.conclusion {{ background-color: #e1f5fe; color: #0277bd; }}
                .tag.summary {{ background-color: #fce4ec; color: #c2185b; }}
                .tag.introduction {{ background-color: #f3e5f5; color: #7b1fa2; }}
                
                .prompt-content-section {{
                    margin-top: 30px;
                }}
                
                .section-title {{
                    color: #495057;
                    font-size: 1.3em;
                    font-weight: 600;
                    margin-bottom: 15px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .prompt-content {{
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 10px;
                    padding: 20px;
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 0.95em;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    position: relative;
                    max-height: 600px;
                    overflow-y: auto;
                }}
                
                .prompt-content::-webkit-scrollbar {{
                    width: 10px;
                }}
                
                .prompt-content::-webkit-scrollbar-track {{
                    background: #f1f1f1;
                    border-radius: 5px;
                }}
                
                .prompt-content::-webkit-scrollbar-thumb {{
                    background: #888;
                    border-radius: 5px;
                }}
                
                .prompt-content::-webkit-scrollbar-thumb:hover {{
                    background: #555;
                }}
                
                .copy-button {{
                    position: absolute;
                    top: 15px;
                    right: 15px;
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 20px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 0.9em;
                    font-weight: 500;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }}
                
                .copy-button:hover {{
                    background-color: #2980b9;
                    transform: translateY(-1px);
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
                }}
                
                .copy-button.copied {{
                    background-color: #27ae60;
                }}
                
                .copy-icon {{
                    width: 16px;
                    height: 16px;
                }}
                
                .stats {{
                    display: flex;
                    gap: 30px;
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .stat {{
                    text-align: center;
                    flex: 1;
                }}
                
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #3498db;
                    display: block;
                }}
                
                .stat-label {{
                    font-size: 0.9em;
                    color: #6c757d;
                    margin-top: 5px;
                }}
                
                @media (max-width: 768px) {{
                    body {{
                        padding: 10px;
                    }}
                    
                    .container {{
                        padding: 20px;
                    }}
                    
                    h1 {{
                        font-size: 1.8em;
                        flex-direction: column;
                        align-items: flex-start;
                        gap: 10px;
                    }}
                    
                    .stats {{
                        flex-direction: column;
                        gap: 15px;
                    }}
                    
                    .copy-button {{
                        padding: 6px 15px;
                        font-size: 0.85em;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="prompt-header">
                    <h1>
                        🎯 {prompt.name}
                        <span class="prompt-version">v{prompt.version}</span>
                    </h1>
                    {f'<p class="prompt-description">{prompt.description}</p>' if prompt.description else ""}
                    {tag_html}
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <span class="stat-value">{len(prompt.content.split())}</span>
                        <span class="stat-label">Words</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value">{len(prompt.content)}</span>
                        <span class="stat-label">Characters</span>
                    </div>
                    <div class="stat">
                        <span class="stat-value">{len(prompt.content.splitlines())}</span>
                        <span class="stat-label">Lines</span>
                    </div>
                </div>
                
                <div class="prompt-content-section">
                    <h2 class="section-title">📝 Prompt Content</h2>
                    <div class="prompt-content" id="promptContent">
                        <button class="copy-button" onclick="copyToClipboard()" id="copyButton">
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
