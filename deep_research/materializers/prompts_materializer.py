"""Materializer for PromptsBundle with custom HTML visualization.

This module provides a materializer that creates beautiful HTML visualizations
for prompt bundles in the ZenML dashboard.
"""

import os
from typing import Dict

from utils.prompt_models import PromptsBundle
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class PromptsBundleMaterializer(PydanticMaterializer):
    """Materializer for PromptsBundle with custom visualization."""

    ASSOCIATED_TYPES = (PromptsBundle,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: PromptsBundle
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the PromptsBundle.

        Args:
            data: The PromptsBundle to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization
        visualization_path = os.path.join(self.uri, "prompts_bundle.html")

        # Create HTML content
        html_content = self._generate_visualization_html(data)

        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, bundle: PromptsBundle) -> str:
        """Generate HTML visualization for the prompts bundle.

        Args:
            bundle: The PromptsBundle to visualize

        Returns:
            HTML string
        """
        # Create HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prompts Bundle - {bundle.created_at}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1400px;
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
                
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 3px solid #e1e8ed;
                }}
                
                h1 {{
                    color: #2c3e50;
                    margin: 0 0 10px 0;
                    font-size: 2.5em;
                }}
                
                .metadata {{
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
                
                .stats {{
                    display: flex;
                    justify-content: center;
                    gap: 30px;
                    margin: 20px 0;
                }}
                
                .stat {{
                    text-align: center;
                    padding: 15px 25px;
                    background-color: #f8f9fa;
                    border-radius: 10px;
                    border: 1px solid #e9ecef;
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
                }}
                
                .prompts-grid {{
                    display: grid;
                    gap: 20px;
                    margin-top: 30px;
                }}
                
                .prompt-card {{
                    background-color: #ffffff;
                    border: 1px solid #e1e8ed;
                    border-radius: 12px;
                    padding: 20px;
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }}
                
                .prompt-card:hover {{
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                    transform: translateY(-2px);
                }}
                
                .prompt-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    margin-bottom: 15px;
                }}
                
                .prompt-title {{
                    font-size: 1.3em;
                    color: #2c3e50;
                    font-weight: 600;
                    margin: 0;
                    flex: 1;
                }}
                
                .prompt-version {{
                    background-color: #e3f2fd;
                    color: #1976d2;
                    padding: 4px 10px;
                    border-radius: 15px;
                    font-size: 0.85em;
                    font-weight: 500;
                }}
                
                .prompt-description {{
                    color: #6c757d;
                    font-size: 0.95em;
                    margin-bottom: 15px;
                    font-style: italic;
                }}
                
                .prompt-tags {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin-bottom: 15px;
                }}
                
                .tag {{
                    background-color: #f0f0f0;
                    color: #555;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-size: 0.8em;
                }}
                
                .tag.search {{ background-color: #fff3cd; color: #856404; }}
                .tag.synthesis {{ background-color: #d4edda; color: #155724; }}
                .tag.analysis {{ background-color: #d1ecf1; color: #0c5460; }}
                .tag.reflection {{ background-color: #f8d7da; color: #721c24; }}
                .tag.report {{ background-color: #e2e3e5; color: #383d41; }}
                
                .prompt-content {{
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 15px;
                    margin-top: 15px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 0.9em;
                    line-height: 1.5;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    max-height: 300px;
                    overflow-y: auto;
                    position: relative;
                }}
                
                .prompt-content::-webkit-scrollbar {{
                    width: 8px;
                }}
                
                .prompt-content::-webkit-scrollbar-track {{
                    background: #f1f1f1;
                    border-radius: 4px;
                }}
                
                .prompt-content::-webkit-scrollbar-thumb {{
                    background: #888;
                    border-radius: 4px;
                }}
                
                .prompt-content::-webkit-scrollbar-thumb:hover {{
                    background: #555;
                }}
                
                .expand-button {{
                    position: absolute;
                    bottom: 10px;
                    right: 10px;
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 5px 15px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 0.85em;
                    opacity: 0.9;
                    transition: opacity 0.3s ease;
                }}
                
                .expand-button:hover {{
                    opacity: 1;
                }}
                
                .prompt-content.expanded {{
                    max-height: none;
                }}
                
                .copy-button {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 0.8em;
                    opacity: 0.7;
                    transition: opacity 0.3s ease;
                }}
                
                .copy-button:hover {{
                    opacity: 1;
                }}
                
                .copy-button.copied {{
                    background-color: #28a745;
                }}
                
                .search-box {{
                    margin: 20px 0;
                    position: relative;
                }}
                
                .search-input {{
                    width: 100%;
                    padding: 12px 20px;
                    border: 2px solid #e1e8ed;
                    border-radius: 25px;
                    font-size: 1em;
                    outline: none;
                    transition: border-color 0.3s ease;
                }}
                
                .search-input:focus {{
                    border-color: #3498db;
                }}
                
                .no-results {{
                    text-align: center;
                    color: #6c757d;
                    padding: 40px;
                    font-style: italic;
                }}
                
                @media (max-width: 768px) {{
                    .stats {{
                        flex-direction: column;
                        gap: 10px;
                    }}
                    
                    .prompt-header {{
                        flex-direction: column;
                        gap: 10px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 Prompts Bundle</h1>
                    <div class="metadata">
                        <p>Pipeline Version: <strong>{bundle.pipeline_version}</strong></p>
                        <p>Created: <strong>{bundle.created_at}</strong></p>
                    </div>
                    <div class="stats">
                        <div class="stat">
                            <span class="stat-value">{len(bundle.list_all_prompts())}</span>
                            <span class="stat-label">Total Prompts</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value">{len([p for p in bundle.list_all_prompts().values() if p.tags])}</span>
                            <span class="stat-label">Tagged Prompts</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value">{len(bundle.custom_prompts)}</span>
                            <span class="stat-label">Custom Prompts</span>
                        </div>
                    </div>
                </div>
                
                <div class="search-box">
                    <input type="text" class="search-input" id="searchInput" placeholder="Search prompts by name, description, or content...">
                </div>
                
                <div class="prompts-grid" id="promptsGrid">
        """

        # Add each prompt
        prompts = [
            (
                "search_query_prompt",
                bundle.search_query_prompt,
                ["search", "query"],
            ),
            (
                "query_decomposition_prompt",
                bundle.query_decomposition_prompt,
                ["analysis", "decomposition"],
            ),
            (
                "synthesis_prompt",
                bundle.synthesis_prompt,
                ["synthesis", "integration"],
            ),
            (
                "viewpoint_analysis_prompt",
                bundle.viewpoint_analysis_prompt,
                ["analysis", "viewpoint"],
            ),
            (
                "reflection_prompt",
                bundle.reflection_prompt,
                ["reflection", "critique"],
            ),
            (
                "additional_synthesis_prompt",
                bundle.additional_synthesis_prompt,
                ["synthesis", "enhancement"],
            ),
            (
                "conclusion_generation_prompt",
                bundle.conclusion_generation_prompt,
                ["report", "conclusion"],
            ),
        ]

        for prompt_type, prompt, default_tags in prompts:
            # Use provided tags or default tags
            tags = prompt.tags if prompt.tags else default_tags

            html += f"""
                    <div class="prompt-card" data-prompt-name="{prompt.name}">
                        <div class="prompt-header">
                            <h3 class="prompt-title">{prompt.name}</h3>
                            <span class="prompt-version">v{prompt.version}</span>
                        </div>
                        {f'<p class="prompt-description">{prompt.description}</p>' if prompt.description else ""}
                        <div class="prompt-tags">
                            {"".join([f'<span class="tag {tag}">{tag}</span>' for tag in tags])}
                        </div>
                        <div class="prompt-content" id="content-{prompt_type}">
                            <button class="copy-button" onclick="copyToClipboard('{prompt_type}')">Copy</button>
                            {self._escape_html(prompt.content)}
                        </div>
                        <button class="expand-button" onclick="toggleExpand('{prompt_type}')">Expand</button>
                    </div>
            """

        # Add custom prompts if any
        for name, prompt in bundle.custom_prompts.items():
            tags = prompt.tags if prompt.tags else ["custom"]
            html += f"""
                    <div class="prompt-card" data-prompt-name="{prompt.name}">
                        <div class="prompt-header">
                            <h3 class="prompt-title">{prompt.name}</h3>
                            <span class="prompt-version">v{prompt.version}</span>
                        </div>
                        {f'<p class="prompt-description">{prompt.description}</p>' if prompt.description else ""}
                        <div class="prompt-tags">
                            {"".join([f'<span class="tag {tag}">{tag}</span>' for tag in tags])}
                        </div>
                        <div class="prompt-content" id="content-custom-{name}">
                            <button class="copy-button" onclick="copyToClipboard('custom-{name}')">Copy</button>
                            {self._escape_html(prompt.content)}
                        </div>
                        <button class="expand-button" onclick="toggleExpand('custom-{name}')">Expand</button>
                    </div>
            """

        html += """
                </div>
                <div class="no-results" id="noResults" style="display: none;">
                    No prompts found matching your search criteria.
                </div>
            </div>
            
            <script>
                function toggleExpand(promptType) {
                    const content = document.getElementById('content-' + promptType);
                    const button = content.nextElementSibling;
                    
                    if (content.classList.contains('expanded')) {
                        content.classList.remove('expanded');
                        button.textContent = 'Expand';
                    } else {
                        content.classList.add('expanded');
                        button.textContent = 'Collapse';
                    }
                }
                
                function copyToClipboard(promptType) {
                    const content = document.getElementById('content-' + promptType);
                    const text = content.textContent.replace('Copy', '').trim();
                    
                    navigator.clipboard.writeText(text).then(() => {
                        const button = content.querySelector('.copy-button');
                        button.textContent = 'Copied!';
                        button.classList.add('copied');
                        
                        setTimeout(() => {
                            button.textContent = 'Copy';
                            button.classList.remove('copied');
                        }, 2000);
                    });
                }
                
                // Search functionality
                const searchInput = document.getElementById('searchInput');
                const promptCards = document.querySelectorAll('.prompt-card');
                const noResults = document.getElementById('noResults');
                
                searchInput.addEventListener('input', (e) => {
                    const searchTerm = e.target.value.toLowerCase();
                    let hasResults = false;
                    
                    promptCards.forEach(card => {
                        const text = card.textContent.toLowerCase();
                        if (text.includes(searchTerm)) {
                            card.style.display = 'block';
                            hasResults = true;
                        } else {
                            card.style.display = 'none';
                        }
                    });
                    
                    noResults.style.display = hasResults ? 'none' : 'block';
                });
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
