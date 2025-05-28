"""Materializer for TracingMetadata with custom visualization."""

import os
from typing import Dict

from utils.css_utils import (
    create_stat_card,
    get_card_class,
    get_grid_class,
    get_shared_css_tag,
    get_table_class,
)
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

        # Build stats cards HTML
        stats_html = f"""
        <div class="{get_grid_class("metrics")}">
            <div class="dr-stat-card">
                <div class="dr-stat-label">Pipeline Run</div>
                <div class="dr-stat-value" style="font-size: 16px;">{metadata.pipeline_run_name}</div>
            </div>
            {create_stat_card(f"${metadata.total_cost:.4f}", "LLM Cost")}
            {create_stat_card(f"{metadata.total_tokens:,}", "Total Tokens")}
            {create_stat_card(metadata.formatted_latency, "Duration")}
        </div>
        """

        token_stats_html = f"""
        <div class="{get_grid_class("metrics")}">
            {create_stat_card(f"{metadata.total_input_tokens:,}", "Input Tokens")}
            {create_stat_card(f"{metadata.total_output_tokens:,}", "Output Tokens")}
            {create_stat_card(metadata.observation_count, "Observations")}
            {create_stat_card(f"${avg_cost_per_token:.6f}", "Avg Cost per Token")}
        </div>
        """

        # Base structure for the HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pipeline Tracing Metadata</title>
            {get_shared_css_tag()}
            <style>
                /* Component-specific styles */
                .metric-breakdown-section {{
                    margin-top: var(--spacing-lg);
                }}
                
                .dr-tag-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: var(--spacing-xs);
                    margin-top: var(--spacing-sm);
                }}
            </style>
        </head>
        <body>
            <div class="dr-container">
                <h1>Pipeline Tracing Metadata</h1>
                
                {stats_html}
                
                <h2>Token Usage</h2>
                {token_stats_html}
                
                <h2>Model Usage Breakdown</h2>
                <table class="{get_table_class(striped=True)}">
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
            html += f"""
                <h2>Cost Analysis by Prompt Type</h2>
                
                <!-- Chart.js library -->
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                
                <!-- Cost by prompt type bar chart -->
                <div class="{get_card_class()}">
                    <h3>Cost Distribution</h3>
                    <div class="dr-chart-container">
                        <canvas id="promptCostChart"></canvas>
                    </div>
                </div>
                
                <!-- Token usage by prompt type stacked bar chart -->
                <div class="{get_card_class()}">
                    <h3>Token Usage</h3>
                    <div class="dr-chart-container">
                        <canvas id="promptTokenChart"></canvas>
                    </div>
                </div>
                
                <!-- Prompt efficiency table -->
                <h3>Prompt Type Efficiency</h3>
                <table class="{get_table_class(striped=True)}">
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

            # Build search provider cards
            search_cards = ""
            for provider, cost in metadata.search_costs.items():
                if cost > 0:
                    query_count = metadata.search_queries_count.get(
                        provider, 0
                    )
                    avg_cost_per_query = (
                        cost / query_count if query_count > 0 else 0
                    )
                    search_cards += f"""
                    <div class="dr-stat-card">
                        <div class="dr-stat-label">{provider.upper()} Search</div>
                        <div class="dr-stat-value">${cost:.4f}</div>
                        <div class="dr-text-muted" style="font-size: 0.75rem; margin-top: 5px;">
                            {query_count} queries • ${avg_cost_per_query:.4f}/query
                        </div>
                    </div>
                    """

            search_cards += f"""
                <div class="dr-stat-card" style="border-color: var(--color-warning);">
                    <div class="dr-stat-label">Total Search Cost</div>
                    <div class="dr-stat-value" style="color: var(--color-warning);">${total_search_cost:.4f}</div>
                    <div class="dr-text-muted" style="font-size: 0.75rem; margin-top: 5px;">
                        {sum(metadata.search_queries_count.values())} total queries
                    </div>
                </div>
            """

            html += f"""
                <h2>Search Provider Costs</h2>
                <div class="{get_grid_class("metrics")}">
                    {search_cards}
                </div>
                
                <h2>Combined Cost Summary</h2>
                <div class="{get_grid_class("stats")}">
                    <div class="dr-stat-card">
                        <div class="dr-stat-label">LLM Cost</div>
                        <div class="dr-stat-value">${metadata.total_cost:.4f}</div>
                        <div class="dr-text-muted" style="font-size: 0.75rem;">
                            {(metadata.total_cost / total_combined_cost * 100):.1f}% of total
                        </div>
                    </div>
                    <div class="dr-stat-card">
                        <div class="dr-stat-label">Search Cost</div>
                        <div class="dr-stat-value">${total_search_cost:.4f}</div>
                        <div class="dr-text-muted" style="font-size: 0.75rem;">
                            {(total_search_cost / total_combined_cost * 100):.1f}% of total
                        </div>
                    </div>
                    <div class="dr-stat-card" style="border: 2px solid var(--color-danger);">
                        <div class="dr-stat-label">Total Pipeline Cost</div>
                        <div class="dr-stat-value" style="color: var(--color-danger);">${total_combined_cost:.4f}</div>
                    </div>
                </div>
                
                <div class="{get_card_class()}">
                    <h3>Cost Breakdown Chart</h3>
                    <div class="dr-chart-container" style="max-width: 400px; margin: 0 auto;">
                        <canvas id="costBreakdownChart"></canvas>
                    </div>
                </div>
                <script>
                    var ctx = document.getElementById('costBreakdownChart').getContext('2d');
                    var totalCombinedCost = {total_combined_cost:.4f};
                    var llmCost = {metadata.total_cost:.4f};
                    var searchCost = {total_search_cost:.4f};
                    var costBreakdownChart = new Chart(ctx, {{
                        type: 'doughnut',
                        data: {{
                            labels: ['LLM Cost', 'Search Cost'],
                            datasets: [{{
                                data: [llmCost, searchCost],
                                backgroundColor: ['#3498db', '#e67e22'],
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: {{
                                legend: {{
                                    position: 'bottom'
                                }},
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            var label = context.label || '';
                                            var value = context.parsed || 0;
                                            var percentage = ((value / totalCombinedCost) * 100).toFixed(1);
                                            return label + ': $' + value.toFixed(4) + ' (' + percentage + '%)';
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }});
                </script>
            """

        # Add trace metadata
        if metadata.trace_tags or metadata.trace_metadata:
            html += f"""
                <h2>Trace Information</h2>
                <div class="{get_card_class()}">
            """

            if metadata.trace_tags:
                html += """
                    <h3>Tags</h3>
                    <div class="dr-tag-container">
                """
                for tag in metadata.trace_tags:
                    html += (
                        f'<span class="dr-tag dr-tag--primary">{tag}</span>'
                    )
                html += """
                    </div>
                """

            if metadata.trace_metadata:
                html += """
                    <h3>Metadata</h3>
                    <pre class="dr-code">
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
                <div class="dr-timestamp">
                    <p><strong>Trace ID:</strong> {metadata.trace_id}</p>
                    <p><strong>Pipeline Run ID:</strong> {metadata.pipeline_run_id}</p>
                    <p><strong>Collected at:</strong> {collection_time}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html
