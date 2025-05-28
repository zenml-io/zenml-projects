"""Materializer for SearchData with cost breakdown charts and search results visualization."""

import json
import os
from typing import Dict

from utils.css_utils import (
    create_stat_card,
    get_card_class,
    get_grid_class,
    get_shared_css_tag,
    get_table_class,
)
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
            <div class="dr-section dr-mb-lg">
                <h3>{sub_q}</h3>
                <div class="dr-text-secondary dr-mb-md">{len(results)} results found</div>
                <div class="results-list">
            """

            for i, result in enumerate(results[:5]):  # Show first 5 results
                results_html += f"""
                <div class="dr-result-item">
                    <div class="dr-result-title">{result.title or "Untitled"}</div>
                    <div class="dr-result-snippet">{result.snippet or result.content[:200]}...</div>
                    <a href="{result.url}" target="_blank" class="dr-result-link">View Source</a>
                </div>
                """

            if len(results) > 5:
                results_html += f'<div class="dr-text-center dr-text-muted dr-mt-md">... and {len(results) - 5} more results</div>'

            results_html += """
                </div>
            </div>
            """

        if not results_html:
            results_html = '<div class="dr-empty">No search results yet</div>'

        # Calculate total cost
        total_cost = sum(data.search_costs.values())

        # Build stats cards
        stats_html = f"""
        <div class="{get_grid_class("stats")}">
            {create_stat_card(data.total_searches, "Total Searches")}
            {create_stat_card(len(data.search_results), "Sub-Questions")}
            {create_stat_card(sum(len(results) for results in data.search_results.values()), "Total Results")}
            {create_stat_card(f"${total_cost:.4f}", "Total Cost")}
        </div>
        """

        # Build cost table rows
        cost_table_rows = "".join(
            f"""
            <tr>
                <td>{provider}</td>
                <td>${cost:.4f}</td>
                <td>{(cost / total_cost * 100 if total_cost > 0 else 0):.1f}%</td>
            </tr>
            """
            for provider, cost in data.search_costs.items()
        )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Search Data Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            {get_shared_css_tag()}
        </head>
        <body>
            <div class="dr-container dr-container--wide">
                <div class="dr-header-card">
                    <h1>Search Data Analysis</h1>
                    {stats_html}
                </div>
                
                <div class="{get_card_class()}">
                    <h2>Cost Analysis</h2>
                    
                    <div class="dr-chart-container">
                        <canvas id="costChart"></canvas>
                    </div>
                    
                    <div class="dr-mt-lg">
                        <h3>Cost Breakdown by Provider</h3>
                        <table class="{get_table_class()}">
                            <thead>
                                <tr>
                                    <th>Provider</th>
                                    <th>Cost</th>
                                    <th>Percentage</th>
                                </tr>
                            </thead>
                            <tbody>
                                {cost_table_rows}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="{get_card_class()}">
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
