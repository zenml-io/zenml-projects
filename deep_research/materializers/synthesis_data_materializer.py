"""Materializer for SynthesisData with confidence metrics and synthesis quality visualization."""

import os
from typing import Dict

from utils.pydantic_models import SynthesisData
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class SynthesisDataMaterializer(PydanticMaterializer):
    """Materializer for SynthesisData with quality metrics visualization."""

    ASSOCIATED_TYPES = (SynthesisData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: SynthesisData
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the SynthesisData.

        Args:
            data: The SynthesisData to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualization_path = os.path.join(self.uri, "synthesis_data.html")
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, data: SynthesisData) -> str:
        """Generate HTML visualization for the synthesis data.

        Args:
            data: The SynthesisData to visualize

        Returns:
            HTML string
        """
        # Count confidence levels
        confidence_counts = {"high": 0, "medium": 0, "low": 0}
        for info in data.synthesized_info.values():
            confidence_counts[info.confidence_level] += 1

        # Create synthesis cards HTML
        synthesis_html = ""
        for sub_q, info in data.synthesized_info.items():
            confidence_color = {
                "high": "#2dce89",
                "medium": "#ffd600",
                "low": "#f5365c",
            }.get(info.confidence_level, "#666")

            sources_html = ""
            if info.key_sources:
                sources_html = (
                    "<div class='sources'><strong>Key Sources:</strong><ul>"
                )
                for source in info.key_sources[:3]:  # Show first 3 sources
                    sources_html += f"<li>{source}</li>"
                if len(info.key_sources) > 3:
                    sources_html += (
                        f"<li>... and {len(info.key_sources) - 3} more</li>"
                    )
                sources_html += "</ul></div>"

            gaps_html = ""
            if info.information_gaps:
                gaps_html = f"""
                <div class='gaps'>
                    <strong>Information Gaps:</strong>
                    <p>{info.information_gaps}</p>
                </div>
                """

            improvements_html = ""
            if info.improvements:
                improvements_html = "<div class='improvements'><strong>Suggested Improvements:</strong><ul>"
                for imp in info.improvements:
                    improvements_html += f"<li>{imp}</li>"
                improvements_html += "</ul></div>"

            # Check if this has enhanced version
            enhanced_badge = ""
            enhanced_section = ""
            if sub_q in data.enhanced_info:
                enhanced_badge = '<span class="enhanced-badge">Enhanced</span>'
                enhanced_info = data.enhanced_info[sub_q]
                enhanced_section = f"""
                <div class="enhanced-section">
                    <h4>Enhanced Answer</h4>
                    <p>{enhanced_info.synthesized_answer}</p>
                    <div class="confidence-badge" style="background-color: {
                    {
                        "high": "#2dce89",
                        "medium": "#ffd600",
                        "low": "#f5365c",
                    }.get(enhanced_info.confidence_level, "#666")
                }">
                        Confidence: {enhanced_info.confidence_level.upper()}
                    </div>
                </div>
                """

            synthesis_html += f"""
            <div class="synthesis-card">
                <div class="card-header">
                    <h3>{sub_q}</h3>
                    {enhanced_badge}
                </div>
                
                <div class="original-section">
                    <h4>Original Synthesis</h4>
                    <p class="synthesized-answer">{info.synthesized_answer}</p>
                    
                    <div class="confidence-badge" style="background-color: {confidence_color}">
                        Confidence: {info.confidence_level.upper()}
                    </div>
                    
                    {sources_html}
                    {gaps_html}
                    {improvements_html}
                </div>
                
                {enhanced_section}
            </div>
            """

        if not synthesis_html:
            synthesis_html = '<div class="no-synthesis">No synthesis data available yet</div>'

        # Calculate statistics
        total_syntheses = len(data.synthesized_info)
        total_enhanced = len(data.enhanced_info)
        avg_sources = sum(
            len(info.key_sources) for info in data.synthesized_info.values()
        ) / max(total_syntheses, 1)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synthesis Data Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f0f2f5;
                    color: #333;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                
                .header {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
                    margin-bottom: 30px;
                }}
                
                h1 {{
                    margin: 0 0 20px 0;
                    color: #2c3e50;
                    font-size: 32px;
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .stat-card {{
                    background: white;
                    border-radius: 12px;
                    padding: 25px;
                    text-align: center;
                    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
                    transition: transform 0.3s ease;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
                }}
                
                .stat-value {{
                    font-size: 36px;
                    font-weight: bold;
                    color: #5e72e4;
                    margin-bottom: 5px;
                }}
                
                .stat-label {{
                    color: #666;
                    font-size: 14px;
                }}
                
                .confidence-chart-section {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
                    margin-bottom: 30px;
                }}
                
                .chart-container {{
                    position: relative;
                    height: 300px;
                    margin: 20px 0;
                }}
                
                .synthesis-section {{
                    margin-bottom: 30px;
                }}
                
                .synthesis-card {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 20px;
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
                }}
                
                .card-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                
                .card-header h3 {{
                    color: #2c3e50;
                    margin: 0;
                    flex: 1;
                }}
                
                .enhanced-badge {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                    text-transform: uppercase;
                }}
                
                .original-section,
                .enhanced-section {{
                    margin-bottom: 20px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .enhanced-section {{
                    background: #f0f5ff;
                    border: 2px solid #667eea;
                }}
                
                h4 {{
                    color: #495057;
                    margin: 0 0 15px 0;
                    font-size: 16px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                
                .synthesized-answer {{
                    line-height: 1.8;
                    color: #333;
                    margin-bottom: 15px;
                }}
                
                .confidence-badge {{
                    display: inline-block;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .sources {{
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #e9ecef;
                }}
                
                .sources ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                
                .sources li {{
                    color: #666;
                    margin: 5px 0;
                }}
                
                .gaps {{
                    margin-top: 20px;
                    padding: 15px;
                    background: #fff5f5;
                    border-radius: 8px;
                    border-left: 4px solid #f5365c;
                }}
                
                .gaps p {{
                    margin: 10px 0 0 0;
                    color: #666;
                }}
                
                .improvements {{
                    margin-top: 20px;
                    padding: 15px;
                    background: #f6fff9;
                    border-radius: 8px;
                    border-left: 4px solid #2dce89;
                }}
                
                .improvements ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                
                .improvements li {{
                    color: #666;
                    margin: 5px 0;
                }}
                
                .no-synthesis {{
                    text-align: center;
                    color: #999;
                    font-style: italic;
                    padding: 60px;
                    background: white;
                    border-radius: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Synthesis Quality Analysis</h1>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{total_syntheses}</div>
                        <div class="stat-label">Total Syntheses</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{total_enhanced}</div>
                        <div class="stat-label">Enhanced Syntheses</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{avg_sources:.1f}</div>
                        <div class="stat-label">Avg Sources per Synthesis</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{confidence_counts["high"]}</div>
                        <div class="stat-label">High Confidence</div>
                    </div>
                </div>
                
                <div class="confidence-chart-section">
                    <h2>Confidence Distribution</h2>
                    <div class="chart-container">
                        <canvas id="confidenceChart"></canvas>
                    </div>
                </div>
                
                <div class="synthesis-section">
                    <h2>Synthesized Information</h2>
                    {synthesis_html}
                </div>
            </div>
            
            <script>
                // Confidence distribution bar chart
                const ctx = document.getElementById('confidenceChart').getContext('2d');
                
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: ['High', 'Medium', 'Low'],
                        datasets: [{{
                            label: 'Number of Syntheses',
                            data: [{confidence_counts["high"]}, {confidence_counts["medium"]}, {confidence_counts["low"]}],
                            backgroundColor: ['#2dce89', '#ffd600', '#f5365c'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                ticks: {{
                                    stepSize: 1
                                }}
                            }}
                        }},
                        plugins: {{
                            legend: {{
                                display: false
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return html
