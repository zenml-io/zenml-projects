"""Materializer for SearchData with cost breakdown charts and search results visualization."""

import json
import os
from typing import Dict

from utils.pydantic_models import SearchData
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class SearchDataMaterializer(PydanticMaterializer):
    """Materializer for SearchData with interactive visualizations."""

    ASSOCIATED_TYPES = (SearchData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: SearchData
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the SearchData.

        Args:
            data: The SearchData to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualization_path = os.path.join(self.uri, "search_data.html")
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, data: SearchData) -> str:
        """Generate HTML visualization for the search data.

        Args:
            data: The SearchData to visualize

        Returns:
            HTML string
        """
        # Prepare data for charts
        cost_data = [
            {"provider": k, "cost": v} for k, v in data.search_costs.items()
        ]

        # Create search results HTML
        results_html = ""
        for sub_q, results in data.search_results.items():
            results_html += f"""
            <div class="sub-question-results">
                <h3>{sub_q}</h3>
                <div class="results-count">{len(results)} results found</div>
                <div class="results-list">
            """

            for i, result in enumerate(results[:5]):  # Show first 5 results
                results_html += f"""
                <div class="result-item">
                    <div class="result-title">{result.title or "Untitled"}</div>
                    <div class="result-snippet">{result.snippet or result.content[:200]}...</div>
                    <a href="{result.url}" target="_blank" class="result-link">View Source</a>
                </div>
                """

            if len(results) > 5:
                results_html += f'<div class="more-results">... and {len(results) - 5} more results</div>'

            results_html += """
                </div>
            </div>
            """

        if not results_html:
            results_html = (
                '<div class="no-results">No search results yet</div>'
            )

        # Calculate total cost
        total_cost = sum(data.search_costs.values())

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Search Data Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f7fa;
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
                    margin-top: 20px;
                }}
                
                .stat-card {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;
                    border: 2px solid #e9ecef;
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
                
                .charts-section {{
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
                
                .results-section {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
                }}
                
                .sub-question-results {{
                    margin-bottom: 40px;
                    border-bottom: 1px solid #e9ecef;
                    padding-bottom: 30px;
                }}
                
                .sub-question-results:last-child {{
                    border-bottom: none;
                }}
                
                .sub-question-results h3 {{
                    color: #2c3e50;
                    margin: 0 0 15px 0;
                }}
                
                .results-count {{
                    color: #666;
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
                
                .result-item {{
                    background: #f8f9fa;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border: 1px solid #e9ecef;
                    transition: all 0.3s ease;
                }}
                
                .result-item:hover {{
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                    transform: translateY(-2px);
                }}
                
                .result-title {{
                    font-weight: bold;
                    color: #2c3e50;
                    margin-bottom: 8px;
                }}
                
                .result-snippet {{
                    color: #666;
                    font-size: 14px;
                    line-height: 1.6;
                    margin-bottom: 10px;
                }}
                
                .result-link {{
                    color: #5e72e4;
                    text-decoration: none;
                    font-size: 14px;
                    font-weight: 500;
                }}
                
                .result-link:hover {{
                    text-decoration: underline;
                }}
                
                .more-results {{
                    text-align: center;
                    color: #999;
                    font-style: italic;
                    margin-top: 15px;
                }}
                
                .no-results {{
                    text-align: center;
                    color: #999;
                    font-style: italic;
                    padding: 40px;
                }}
                
                .cost-details {{
                    margin-top: 30px;
                }}
                
                .cost-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                
                .cost-table th,
                .cost-table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #e9ecef;
                }}
                
                .cost-table th {{
                    background: #f8f9fa;
                    font-weight: 600;
                    color: #2c3e50;
                }}
                
                .cost-table tr:hover {{
                    background: #f8f9fa;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Search Data Analysis</h1>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{data.total_searches}</div>
                            <div class="stat-label">Total Searches</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{
            len(data.search_results)
        }</div>
                            <div class="stat-label">Sub-Questions</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{
            sum(len(results) for results in data.search_results.values())
        }</div>
                            <div class="stat-label">Total Results</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">${total_cost:.4f}</div>
                            <div class="stat-label">Total Cost</div>
                        </div>
                    </div>
                </div>
                
                <div class="charts-section">
                    <h2>Cost Analysis</h2>
                    
                    <div class="chart-container">
                        <canvas id="costChart"></canvas>
                    </div>
                    
                    <div class="cost-details">
                        <h3>Cost Breakdown by Provider</h3>
                        <table class="cost-table">
                            <thead>
                                <tr>
                                    <th>Provider</th>
                                    <th>Cost</th>
                                    <th>Percentage</th>
                                </tr>
                            </thead>
                            <tbody>
                                {
            "".join(
                f'''
                                <tr>
                                    <td>{provider}</td>
                                    <td>${cost:.4f}</td>
                                    <td>{(cost / total_cost * 100 if total_cost > 0 else 0):.1f}%</td>
                                </tr>
                                '''
                for provider, cost in data.search_costs.items()
            )
        }
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="results-section">
                    <h2>Search Results</h2>
                    {results_html}
                </div>
            </div>
            
            <script>
                // Cost breakdown pie chart
                const ctx = document.getElementById('costChart').getContext('2d');
                const costData = {json.dumps(cost_data)};
                
                new Chart(ctx, {{
                    type: 'doughnut',
                    data: {{
                        labels: costData.map(d => d.provider),
                        datasets: [{{
                            data: costData.map(d => d.cost),
                            backgroundColor: [
                                '#5e72e4',
                                '#2dce89',
                                '#11cdef',
                                '#f5365c',
                                '#fb6340',
                                '#ffd600'
                            ],
                            borderWidth: 2,
                            borderColor: '#fff'
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'right',
                                labels: {{
                                    padding: 15,
                                    font: {{
                                        size: 14
                                    }}
                                }}
                            }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        const value = context.raw;
                                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                        const percentage = ((value / total) * 100).toFixed(1);
                                        return context.label + ': $' + value.toFixed(4) + ' (' + percentage + '%)';
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return html
