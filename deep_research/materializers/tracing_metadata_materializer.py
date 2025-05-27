"""Materializer for TracingMetadata with custom visualization."""

import os
from typing import Dict

from utils.pydantic_models import TracingMetadata
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class TracingMetadataMaterializer(PydanticMaterializer):
    """Materializer for the TracingMetadata class with visualizations."""

    ASSOCIATED_TYPES = (TracingMetadata,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: TracingMetadata
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the TracingMetadata.

        Args:
            data: The TracingMetadata to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        # Generate an HTML visualization
        visualization_path = os.path.join(self.uri, "tracing_metadata.html")

        # Create HTML content
        html_content = self._generate_visualization_html(data)

        # Write the HTML content to a file
        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        # Return the visualization path and type
        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, metadata: TracingMetadata) -> str:
        """Generate HTML visualization for the tracing metadata.

        Args:
            metadata: The TracingMetadata to visualize

        Returns:
            HTML string
        """
        # Calculate some derived values
        avg_cost_per_token = metadata.total_cost / max(
            metadata.total_tokens, 1
        )

        # Base structure for the HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pipeline Tracing Metadata</title>
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
                }}
                h1 {{
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    border-left: 4px solid #3498db;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin: 5px 0;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 14px;
                }}
                .cost-breakdown {{
                    margin: 20px 0;
                }}
                .model-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .model-table th, .model-table td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .model-table th {{
                    background-color: #3498db;
                    color: white;
                }}
                .model-table tr:hover {{
                    background-color: #f5f5f5;
                }}
                .step-breakdown {{
                    margin: 20px 0;
                }}
                .step-item {{
                    background-color: #f8f9fa;
                    border-left: 3px solid #e67e22;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                }}
                .metadata-info {{
                    font-size: 12px;
                    color: #7f8c8d;
                    margin-top: 20px;
                    padding-top: 10px;
                    border-top: 1px dashed #ddd;
                }}
                .tag {{
                    display: inline-block;
                    background-color: #e1f5fe;
                    color: #0277bd;
                    padding: 3px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    margin: 2px;
                }}
                .progress-bar {{
                    background-color: #ecf0f1;
                    border-radius: 10px;
                    height: 20px;
                    overflow: hidden;
                    position: relative;
                }}
                .progress-fill {{
                    height: 100%;
                    background-color: #3498db;
                    display: flex;
                    align-items: center;
                    padding: 0 10px;
                    color: white;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Pipeline Tracing Metadata</h1>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Pipeline Run</div>
                        <div class="metric-value" style="font-size: 16px;">{metadata.pipeline_run_name}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">LLM Cost</div>
                        <div class="metric-value">${metadata.total_cost:.4f}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Total Tokens</div>
                        <div class="metric-value">{metadata.total_tokens:,}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Duration</div>
                        <div class="metric-value">{metadata.formatted_latency}</div>
                    </div>
                </div>
                
                <h2>Token Usage</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Input Tokens</div>
                        <div class="metric-value">{metadata.total_input_tokens:,}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Output Tokens</div>
                        <div class="metric-value">{metadata.total_output_tokens:,}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Observations</div>
                        <div class="metric-value">{metadata.observation_count}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Avg Cost per Token</div>
                        <div class="metric-value">${avg_cost_per_token:.6f}</div>
                    </div>
                </div>
                
                <h2>Model Usage Breakdown</h2>
                <table class="model-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Input Tokens</th>
                            <th>Output Tokens</th>
                            <th>Total Tokens</th>
                            <th>Cost</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add model breakdown
        for model in metadata.models_used:
            tokens = metadata.model_token_breakdown.get(model, {})
            cost = metadata.cost_breakdown_by_model.get(model, 0.0)
            html += f"""
                        <tr>
                            <td>{model}</td>
                            <td>{tokens.get("input_tokens", 0):,}</td>
                            <td>{tokens.get("output_tokens", 0):,}</td>
                            <td>{tokens.get("total_tokens", 0):,}</td>
                            <td>${cost:.4f}</td>
                        </tr>
            """

        html += """
                    </tbody>
                </table>
        """

        # Add prompt-level metrics visualization
        if metadata.prompt_metrics:
            html += """
                <h2>Cost Analysis by Prompt Type</h2>
                
                <!-- Chart.js library -->
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                
                <!-- Cost by prompt type bar chart -->
                <div style="max-width: 800px; margin: 20px auto;">
                    <canvas id="promptCostChart"></canvas>
                </div>
                
                <!-- Token usage by prompt type stacked bar chart -->
                <div style="max-width: 800px; margin: 20px auto;">
                    <canvas id="promptTokenChart"></canvas>
                </div>
                
                <!-- Prompt efficiency table -->
                <h3>Prompt Type Efficiency</h3>
                <table class="model-table">
                    <thead>
                        <tr>
                            <th>Prompt Type</th>
                            <th>Total Cost</th>
                            <th>Calls</th>
                            <th>Avg $/Call</th>
                            <th>% of Total</th>
                            <th>Input Tokens</th>
                            <th>Output Tokens</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            # Add prompt metrics rows
            for metric in metadata.prompt_metrics:
                # Format prompt type name nicely
                prompt_type_display = metric.prompt_type.replace(
                    "_", " "
                ).title()
                html += f"""
                        <tr>
                            <td>{prompt_type_display}</td>
                            <td>${metric.total_cost:.4f}</td>
                            <td>{metric.call_count}</td>
                            <td>${metric.avg_cost_per_call:.4f}</td>
                            <td>{metric.percentage_of_total_cost:.1f}%</td>
                            <td>{metric.input_tokens:,}</td>
                            <td>{metric.output_tokens:,}</td>
                        </tr>
                """

            html += """
                    </tbody>
                </table>
                
                <script>
                    // Data for the cost chart
                    const promptLabels = [
            """

            # Add labels and data for charts
            labels = [
                metric.prompt_type.replace("_", " ").title()
                for metric in metadata.prompt_metrics
            ]
            costs = [metric.total_cost for metric in metadata.prompt_metrics]
            input_tokens = [
                metric.input_tokens for metric in metadata.prompt_metrics
            ]
            output_tokens = [
                metric.output_tokens for metric in metadata.prompt_metrics
            ]

            html += ", ".join([f'"{label}"' for label in labels])
            html += "];\n"

            html += "const promptCosts = ["
            html += ", ".join([f"{cost:.4f}" for cost in costs])
            html += "];\n"

            html += "const promptInputTokens = ["
            html += ", ".join([str(tokens) for tokens in input_tokens])
            html += "];\n"

            html += "const promptOutputTokens = ["
            html += ", ".join([str(tokens) for tokens in output_tokens])
            html += "];\n"

            html += """
                    // Cost by prompt type chart
                    const costCtx = document.getElementById('promptCostChart').getContext('2d');
                    new Chart(costCtx, {
                        type: 'bar',
                        data: {
                            labels: promptLabels,
                            datasets: [{
                                label: 'Cost ($)',
                                data: promptCosts,
                                backgroundColor: 'rgba(52, 152, 219, 0.8)',
                                borderColor: 'rgba(52, 152, 219, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Cost by Prompt Type'
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            return 'Cost: $' + context.parsed.y.toFixed(4);
                                        }
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    ticks: {
                                        callback: function(value) {
                                            return '$' + value.toFixed(2);
                                        }
                                    }
                                }
                            }
                        }
                    });
                    
                    // Token usage by prompt type chart
                    const tokenCtx = document.getElementById('promptTokenChart').getContext('2d');
                    new Chart(tokenCtx, {
                        type: 'bar',
                        data: {
                            labels: promptLabels,
                            datasets: [
                                {
                                    label: 'Input Tokens',
                                    data: promptInputTokens,
                                    backgroundColor: 'rgba(230, 126, 34, 0.8)',
                                    borderColor: 'rgba(230, 126, 34, 1)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'Output Tokens',
                                    data: promptOutputTokens,
                                    backgroundColor: 'rgba(46, 204, 113, 0.8)',
                                    borderColor: 'rgba(46, 204, 113, 1)',
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Token Usage by Prompt Type'
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            return context.dataset.label + ': ' + context.parsed.y.toLocaleString() + ' tokens';
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    stacked: true
                                },
                                y: {
                                    stacked: true,
                                    beginAtZero: true,
                                    ticks: {
                                        callback: function(value) {
                                            return value.toLocaleString();
                                        }
                                    }
                                }
                            }
                        }
                    });
                </script>
            """

        # Add search cost visualization if available
        if metadata.search_costs and any(metadata.search_costs.values()):
            total_search_cost = sum(metadata.search_costs.values())
            total_combined_cost = metadata.total_cost + total_search_cost

            html += """
                <h2>Search Provider Costs</h2>
                <div class="metric-grid">
            """

            for provider, cost in metadata.search_costs.items():
                if cost > 0:
                    query_count = metadata.search_queries_count.get(
                        provider, 0
                    )
                    avg_cost_per_query = (
                        cost / query_count if query_count > 0 else 0
                    )
                    html += f"""
                        <div class="metric-card">
                            <div class="metric-label">{provider.upper()} Search</div>
                            <div class="metric-value">${cost:.4f}</div>
                            <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                                {query_count} queries • ${avg_cost_per_query:.4f}/query
                            </div>
                        </div>
                    """

            html += (
                f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Search Cost</div>
                        <div class="metric-value" style="color: #e67e22;">${total_search_cost:.4f}</div>
                        <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                            {sum(metadata.search_queries_count.values())} total queries
                        </div>
                    </div>
                </div>
                
                <h2>Combined Cost Summary</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">LLM Cost</div>
                        <div class="metric-value">${metadata.total_cost:.4f}</div>
                        <div style="font-size: 12px; color: #7f8c8d;">
                            {(metadata.total_cost / total_combined_cost * 100):.1f}% of total
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Search Cost</div>
                        <div class="metric-value">${total_search_cost:.4f}</div>
                        <div style="font-size: 12px; color: #7f8c8d;">
                            {(total_search_cost / total_combined_cost * 100):.1f}% of total
                        </div>
                    </div>
                    <div class="metric-card" style="border: 2px solid #e74c3c;">
                        <div class="metric-label">Total Pipeline Cost</div>
                        <div class="metric-value" style="color: #e74c3c; font-size: 24px;">${total_combined_cost:.4f}</div>
                    </div>
                </div>
                
                <h3>Cost Breakdown Chart</h3>
                <div style="position: relative; height: 300px; width: 100%; max-width: 400px; margin: 0 auto;">
                    <canvas id="costBreakdownChart"></canvas>
                </div>
                <script>
                    var ctx = document.getElementById('costBreakdownChart').getContext('2d');
                    var totalCombinedCost = """
                + f"{total_combined_cost:.4f}"
                + """;
                    var llmCost = """
                + f"{metadata.total_cost:.4f}"
                + """;
                    var searchCost = """
                + f"{total_search_cost:.4f}"
                + """;
                    var costBreakdownChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: ['LLM Cost', 'Search Cost'],
                            datasets: [{
                                data: [llmCost, searchCost],
                                backgroundColor: ['#3498db', '#e67e22'],
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: {
                                legend: {
                                    position: 'bottom'
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            var label = context.label || '';
                                            var value = context.parsed || 0;
                                            var percentage = ((value / totalCombinedCost) * 100).toFixed(1);
                                            return label + ': $' + value.toFixed(4) + ' (' + percentage + '%)';
                                        }
                                    }
                                }
                            }
                        }
                    });
                </script>
            """
            )

        # Add trace metadata
        if metadata.trace_tags or metadata.trace_metadata:
            html += """
                <h2>Trace Information</h2>
                <div class="metric-card">
            """

            if metadata.trace_tags:
                html += """
                    <h3>Tags</h3>
                    <div>
                """
                for tag in metadata.trace_tags:
                    html += f'<span class="tag">{tag}</span>'
                html += """
                    </div>
                """

            if metadata.trace_metadata:
                html += """
                    <h3>Metadata</h3>
                    <pre style="background-color: #f8f9fa; padding: 10px; border-radius: 4px;">
                """
                import json

                html += json.dumps(metadata.trace_metadata, indent=2)
                html += """
                    </pre>
                """

            html += """
                </div>
            """

        # Add footer with collection info
        from datetime import datetime

        collection_time = datetime.fromtimestamp(
            metadata.collected_at
        ).strftime("%Y-%m-%d %H:%M:%S")

        html += f"""
                <div class="metadata-info">
                    <p><strong>Trace ID:</strong> {metadata.trace_id}</p>
                    <p><strong>Pipeline Run ID:</strong> {metadata.pipeline_run_id}</p>
                    <p><strong>Collected at:</strong> {collection_time}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html
