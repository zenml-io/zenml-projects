"""Materializer for FinalReport with enhanced interactive report visualization."""

import os
from datetime import datetime
from typing import Dict

from utils.pydantic_models import FinalReport
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class FinalReportMaterializer(PydanticMaterializer):
    """Materializer for FinalReport with interactive report visualization."""

    ASSOCIATED_TYPES = (FinalReport,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: FinalReport
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the FinalReport.

        Args:
            data: The FinalReport to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Save the actual report
        report_path = os.path.join(self.uri, "final_report.html")
        with fileio.open(report_path, "w") as f:
            f.write(data.report_html)

        # Save a wrapper visualization with metadata
        visualization_path = os.path.join(
            self.uri, "report_visualization.html"
        )
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {
            report_path: VisualizationType.HTML,
            visualization_path: VisualizationType.HTML,
        }

    def _generate_visualization_html(self, data: FinalReport) -> str:
        """Generate HTML wrapper visualization for the final report.

        Args:
            data: The FinalReport to visualize

        Returns:
            HTML string
        """
        # Format timestamp
        timestamp = datetime.fromtimestamp(data.generated_at).strftime(
            "%B %d, %Y at %I:%M %p UTC"
        )

        # Extract some statistics from the HTML report if possible
        report_length = len(data.report_html)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Final Research Report - {data.main_query[:50]}...</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                }}
                
                .header {{
                    background: rgba(255, 255, 255, 0.95);
                    padding: 30px;
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                
                h1 {{
                    color: #2c3e50;
                    margin: 0 0 10px 0;
                    font-size: 32px;
                }}
                
                .metadata {{
                    display: flex;
                    gap: 30px;
                    margin-top: 20px;
                    flex-wrap: wrap;
                }}
                
                .meta-item {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    color: #666;
                }}
                
                .meta-icon {{
                    width: 24px;
                    height: 24px;
                    fill: #667eea;
                }}
                
                .query-box {{
                    background: #f8f9fa;
                    border-left: 4px solid #667eea;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 8px;
                }}
                
                .query-label {{
                    font-weight: bold;
                    color: #495057;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    font-size: 12px;
                    letter-spacing: 0.5px;
                }}
                
                .query-text {{
                    color: #333;
                    font-size: 18px;
                    line-height: 1.6;
                }}
                
                .report-frame-container {{
                    flex: 1;
                    background: white;
                    box-shadow: 0 -5px 20px rgba(0, 0, 0, 0.1);
                    position: relative;
                }}
                
                .report-frame {{
                    width: 100%;
                    height: calc(100vh - 250px);
                    border: none;
                    display: block;
                }}
                
                .view-actions {{
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    display: flex;
                    gap: 10px;
                }}
                
                .action-button {{
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    text-decoration: none;
                    display: inline-block;
                }}
                
                .action-button:hover {{
                    background: #5a63d8;
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
                }}
                
                .loading-indicator {{
                    text-align: center;
                    padding: 60px;
                    color: #666;
                }}
                
                @media (max-width: 768px) {{
                    .metadata {{
                        flex-direction: column;
                        gap: 15px;
                    }}
                    
                    .report-frame {{
                        height: calc(100vh - 300px);
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>Final Research Report</h1>
                    
                    <div class="query-box">
                        <div class="query-label">Research Query</div>
                        <div class="query-text">{data.main_query}</div>
                    </div>
                    
                    <div class="metadata">
                        <div class="meta-item">
                            <svg class="meta-icon" viewBox="0 0 24 24">
                                <path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.11 0-1.99.9-1.99 2L3 19c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11zM7 10h5v5H7z"/>
                            </svg>
                            <span>Generated: {timestamp}</span>
                        </div>
                        <div class="meta-item">
                            <svg class="meta-icon" viewBox="0 0 24 24">
                                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
                            </svg>
                            <span>Report Size: {report_length:,} characters</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="report-frame-container">
                <div class="view-actions">
                    <a href="final_report.html" target="_blank" class="action-button">
                        Open in New Tab
                    </a>
                    <button onclick="window.print()" class="action-button">
                        Print Report
                    </button>
                </div>
                
                <div id="loading" class="loading-indicator">
                    Loading report...
                </div>
                
                <iframe 
                    id="reportFrame"
                    class="report-frame" 
                    src="final_report.html"
                    onload="document.getElementById('loading').style.display='none';"
                ></iframe>
            </div>
            
            <script>
                // Auto-resize iframe to content
                function resizeIframe() {{
                    const iframe = document.getElementById('reportFrame');
                    try {{
                        const height = iframe.contentWindow.document.body.scrollHeight;
                        iframe.style.height = height + 'px';
                    }} catch (e) {{
                        // Cross-origin restriction, keep default height
                    }}
                }}
                
                window.addEventListener('load', resizeIframe);
                window.addEventListener('resize', resizeIframe);
            </script>
        </body>
        </html>
        """

        return html
