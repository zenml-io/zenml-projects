"""Materializer for QueryContext with interactive mind map visualization."""

import os
from typing import Dict

from utils.pydantic_models import QueryContext
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class QueryContextMaterializer(PydanticMaterializer):
    """Materializer for QueryContext with mind map visualization."""

    ASSOCIATED_TYPES = (QueryContext,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: QueryContext
    ) -> Dict[str, VisualizationType]:
        """Create and save mind map visualization for the QueryContext.

        Args:
            data: The QueryContext to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualization_path = os.path.join(self.uri, "query_context.html")
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, context: QueryContext) -> str:
        """Generate HTML mind map visualization for the query context.

        Args:
            context: The QueryContext to visualize

        Returns:
            HTML string
        """
        # Create sub-questions HTML
        sub_questions_html = ""
        if context.sub_questions:
            for i, sub_q in enumerate(context.sub_questions, 1):
                sub_questions_html += f"""
                <div class="sub-question">
                    <div class="sub-question-number">{i}</div>
                    <div class="sub-question-text">{sub_q}</div>
                </div>
                """
        else:
            sub_questions_html = '<div class="no-sub-questions">No sub-questions decomposed yet</div>'

        # Format timestamp
        from datetime import datetime

        timestamp = datetime.fromtimestamp(
            context.decomposition_timestamp
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Query Context - {context.main_query[:50]}...</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                
                .container {{
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    padding: 40px;
                    max-width: 1200px;
                    width: 90%;
                    margin: 20px;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                }}
                
                h1 {{
                    color: #333;
                    margin: 0 0 10px 0;
                    font-size: 28px;
                }}
                
                .timestamp {{
                    color: #666;
                    font-size: 14px;
                }}
                
                .mind-map {{
                    position: relative;
                    margin: 40px 0;
                }}
                
                .main-query {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 15px;
                    text-align: center;
                    font-size: 20px;
                    font-weight: bold;
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                    margin-bottom: 40px;
                    position: relative;
                }}
                
                .main-query::after {{
                    content: '';
                    position: absolute;
                    bottom: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 2px;
                    height: 20px;
                    background: #667eea;
                }}
                
                .sub-questions-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    position: relative;
                }}
                
                .sub-questions-container::before {{
                    content: '';
                    position: absolute;
                    top: -20px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 80%;
                    height: 2px;
                    background: linear-gradient(90deg, transparent, #667eea, transparent);
                }}
                
                .sub-question {{
                    background: #f8f9fa;
                    border: 2px solid #e9ecef;
                    border-radius: 10px;
                    padding: 20px;
                    position: relative;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: flex-start;
                    gap: 15px;
                }}
                
                .sub-question:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    border-color: #667eea;
                }}
                
                .sub-question-number {{
                    background: #667eea;
                    color: white;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    flex-shrink: 0;
                }}
                
                .sub-question-text {{
                    color: #333;
                    line-height: 1.6;
                    flex: 1;
                }}
                
                .no-sub-questions {{
                    text-align: center;
                    color: #999;
                    font-style: italic;
                    padding: 40px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .stats {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    margin-top: 30px;
                    display: flex;
                    justify-content: space-around;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                
                .stat {{
                    text-align: center;
                }}
                
                .stat-value {{
                    font-size: 32px;
                    font-weight: bold;
                    color: #667eea;
                }}
                
                .stat-label {{
                    color: #666;
                    font-size: 14px;
                    margin-top: 5px;
                }}
                
                @media (max-width: 768px) {{
                    .sub-questions-container {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Query Decomposition Mind Map</h1>
                    <div class="timestamp">Created: {timestamp}</div>
                </div>
                
                <div class="mind-map">
                    <div class="main-query">
                        {context.main_query}
                    </div>
                    
                    <div class="sub-questions-container">
                        {sub_questions_html}
                    </div>
                </div>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{len(context.sub_questions)}</div>
                        <div class="stat-label">Sub-Questions</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{len(context.main_query.split())}</div>
                        <div class="stat-label">Words in Query</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{sum(len(q.split()) for q in context.sub_questions)}</div>
                        <div class="stat-label">Total Sub-Question Words</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        return html
