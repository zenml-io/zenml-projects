#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from typing import Annotated, Dict, List

import plotly.graph_objects as go
from zenml import get_step_context, log_metadata, step
from zenml.types import HTMLString


def create_plotly_bar_chart(
    labels: List[str],
    scores: List[float],
    title: str,
    alternate_colors: bool = False,
    percentage_scale: bool = False,
    descriptions: Dict[str, str] = None,
) -> go.Figure:
    """
    Create a horizontal bar chart using Plotly.

    Args:
        labels (List[str]): List of labels for the y-axis.
        scores (List[float]): List of scores corresponding to each label.
        title (str): Title of the chart.
        alternate_colors (bool): Whether to alternate colors for the bars.
        percentage_scale (bool): Whether to use a percentage scale (0-100) for the x-axis.
        descriptions (Dict[str, str]): Optional descriptions for hover text.

    Returns:
        go.Figure: Plotly figure object
    """
    # Generate colors for bars
    if alternate_colors:
        colors = [
            "rgba(66, 133, 244, 0.8)"
            if i % 2 == 0
            else "rgba(219, 68, 55, 0.8)"
            for i in range(len(labels))
        ]
    else:
        colors = ["rgba(66, 133, 244, 0.8)" for _ in range(len(labels))]

    # Prepare hover text
    if descriptions:
        hover_text = [
            f"<b>{label}</b><br>Value: {score:.2f}<br>{descriptions.get(label, '')}"
            for label, score in zip(labels, scores)
        ]
    else:
        hover_text = [
            f"<b>{label}</b><br>Value: {score:.2f}"
            for label, score in zip(labels, scores)
        ]

    # Create figure
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=labels,
            x=scores,
            orientation="h",
            marker_color=colors,
            text=[f"{score:.2f}" for score in scores],
            textposition="auto",
            hovertext=hover_text,
            hoverinfo="text",
        )
    )

    # Set layout
    max_value = max(scores) if scores else 5
    xaxis_range = (
        [0, 100] if percentage_scale else [0, max(5, max_value * 1.1)]
    )
    xaxis_title = "Percentage (%)" if percentage_scale else "Score"

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=xaxis_title,
            range=xaxis_range,
            showgrid=True,
            gridcolor="rgba(230, 230, 230, 0.8)",
        ),
        yaxis=dict(
            autorange="reversed",  # Make labels read top-to-bottom
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=max(300, 70 * len(labels)),
        plot_bgcolor="rgba(255, 255, 255, 1)",
    )

    return fig


def generate_evaluation_html(
    pipeline_run_name: str,
    retrieval_labels: List[str],
    retrieval_scores: List[float],
    generation_basic_labels: List[str],
    generation_basic_scores: List[float],
    generation_quality_labels: List[str],
    generation_quality_scores: List[float],
    metrics_metadata: Dict[str, float],
) -> str:
    """
    Generate a comprehensive HTML report with all evaluation visualizations.

    Args:
        pipeline_run_name (str): Name of the pipeline run
        retrieval_labels (List[str]): Labels for retrieval metrics
        retrieval_scores (List[float]): Scores for retrieval metrics
        generation_basic_labels (List[str]): Labels for basic generation metrics
        generation_basic_scores (List[float]): Scores for basic generation metrics
        generation_quality_labels (List[str]): Labels for generation quality metrics
        generation_quality_scores (List[float]): Scores for generation quality metrics
        metrics_metadata (Dict[str, float]): All metrics for displaying in the summary

    Returns:
        str: HTML string containing the interactive dashboard
    """
    # Metric descriptions for hovering
    metric_descriptions = {
        "Small Retrieval Eval Failure Rate": "Percentage of small test cases where retrieval failed to find relevant documents.",
        "Small Retrieval Eval Failure Rate Reranking": "Percentage of small test cases where retrieval with reranking failed to find relevant documents.",
        "Full Retrieval Eval Failure Rate": "Percentage of all test cases where retrieval failed to find relevant documents.",
        "Full Retrieval Eval Failure Rate Reranking": "Percentage of all test cases where retrieval with reranking failed to find relevant documents.",
        "Failure Rate Bad Answers": "Percentage of responses that were factually incorrect or misleading.",
        "Failure Rate Bad Immediate Responses": "Percentage of immediate responses that did not adequately address the query.",
        "Failure Rate Good Responses": "Percentage of responses rated as good by evaluators.",
        "Average Toxicity Score": "Average score measuring harmful, offensive, or inappropriate content (lower is better).",
        "Average Faithfulness Score": "Average score measuring how accurately the response represents the source material (higher is better).",
        "Average Helpfulness Score": "Average score measuring the practical utility of responses to users (higher is better).",
        "Average Relevance Score": "Average score measuring how well responses address the specific query intent (higher is better).",
    }

    # Create individual charts
    retrieval_fig = create_plotly_bar_chart(
        retrieval_labels,
        retrieval_scores,
        f"Retrieval Evaluation Metrics",
        alternate_colors=True,
        descriptions=metric_descriptions,
    )

    generation_basic_fig = create_plotly_bar_chart(
        generation_basic_labels,
        generation_basic_scores,
        f"Basic Generation Metrics",
        percentage_scale=True,
        descriptions=metric_descriptions,
    )

    generation_quality_fig = create_plotly_bar_chart(
        generation_quality_labels,
        generation_quality_scores,
        f"Generation Quality Metrics",
        descriptions=metric_descriptions,
    )

    # Create summary metrics cards
    composite_quality = metrics_metadata.get("composite.overall_quality", 0)
    retrieval_effectiveness = metrics_metadata.get(
        "composite.retrieval_effectiveness", 0
    )

    # Combine into complete HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Results: {pipeline_run_name}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1, h2 {{
                color: #333;
            }}
            h1 {{
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .chart-container {{
                margin-bottom: 30px;
            }}
            .metrics-summary {{
                display: flex;
                justify-content: space-around;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }}
            .metric-card {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 15px;
                min-width: 200px;
                margin: 10px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #1a73e8;
                margin: 10px 0;
            }}
            .description {{
                margin-top: 15px;
                color: #666;
                line-height: 1.5;
                font-size: 14px;
            }}
            .tabs {{
                display: flex;
                margin-bottom: 15px;
                border-bottom: 1px solid #ddd;
            }}
            .tab {{
                padding: 10px 20px;
                cursor: pointer;
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-bottom: none;
                border-radius: 5px 5px 0 0;
                margin-right: 5px;
            }}
            .tab.active {{
                background-color: white;
                border-bottom: 1px solid white;
                margin-bottom: -1px;
                font-weight: bold;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Evaluation Results: {pipeline_run_name}</h1>
            
            <div class="metrics-summary">
                <div class="metric-card">
                    <h3>Overall Quality Score</h3>
                    <div class="metric-value">{composite_quality:.2f}</div>
                    <p>Average of faithfulness, helpfulness, and relevance</p>
                </div>
                <div class="metric-card">
                    <h3>Retrieval Effectiveness</h3>
                    <div class="metric-value">{retrieval_effectiveness:.2f}</div>
                    <p>Average success rate across retrieval tests</p>
                </div>
                <div class="metric-card">
                    <h3>Toxicity</h3>
                    <div class="metric-value">{metrics_metadata.get("quality.toxicity", 0):.2f}</div>
                    <p>Average toxicity score (lower is better)</p>
                </div>
            </div>

            <div class="tabs">
                <div class="tab active" onclick="openTab(event, 'tab-all')">All Metrics</div>
                <div class="tab" onclick="openTab(event, 'tab-retrieval')">Retrieval</div>
                <div class="tab" onclick="openTab(event, 'tab-generation')">Generation</div>
                <div class="tab" onclick="openTab(event, 'tab-quality')">Quality</div>
            </div>

            <div id="tab-all" class="tab-content active">
                <div class="chart-container" id="retrieval-chart"></div>
                <div class="chart-container" id="generation-basic-chart"></div>
                <div class="chart-container" id="generation-quality-chart"></div>
                
                <h2>All Metrics</h2>
                <table style="width:100%; border-collapse: collapse; margin-top: 15px;">
                    <tr style="background-color: #f2f2f2;">
                        <th style="text-align: left; padding: 8px; border: 1px solid #ddd;">Metric</th>
                        <th style="text-align: right; padding: 8px; border: 1px solid #ddd;">Value</th>
                    </tr>
                    {"".join(f'<tr><td style="padding: 8px; border: 1px solid #ddd;">{k}</td><td style="text-align: right; padding: 8px; border: 1px solid #ddd;">{v:.4f}</td></tr>' for k, v in metrics_metadata.items())}
                </table>
            </div>
            
            <div id="tab-retrieval" class="tab-content">
                <div class="chart-container" id="retrieval-chart-tab"></div>
                <div class="description">
                    <h3>About Retrieval Metrics</h3>
                    <p>
                        These metrics measure how effectively the system retrieves relevant documents for answering queries.
                        Lower failure rates indicate better retrieval performance. Reranking shows the impact of the reranking 
                        algorithm on improving retrieval quality.
                    </p>
                </div>
            </div>
            
            <div id="tab-generation" class="tab-content">
                <div class="chart-container" id="generation-basic-chart-tab"></div>
                <div class="description">
                    <h3>About Generation Failure Metrics</h3>
                    <p>
                        These metrics measure different types of failures in response generation:
                        <ul>
                            <li><strong>Bad Answers:</strong> Responses that are factually incorrect or misleading</li>
                            <li><strong>Bad Immediate Responses:</strong> Initial responses that don't address the query adequately</li> 
                            <li><strong>Good Responses:</strong> The percentage of responses rated as good (higher is better)</li>
                        </ul>
                    </p>
                </div>
            </div>
            
            <div id="tab-quality" class="tab-content">
                <div class="chart-container" id="generation-quality-chart-tab"></div>
                <div class="description">
                    <h3>About Quality Metrics</h3>
                    <p>
                        These metrics evaluate the quality of generated responses across different dimensions:
                        <ul>
                            <li><strong>Toxicity:</strong> Measures harmful or inappropriate content (lower is better)</li>
                            <li><strong>Faithfulness:</strong> Measures accuracy to source material (higher is better)</li>
                            <li><strong>Helpfulness:</strong> Measures practical utility to users (higher is better)</li>
                            <li><strong>Relevance:</strong> Measures alignment with query intent (higher is better)</li>
                        </ul>
                    </p>
                </div>
            </div>
        </div>

        <script>
            // Render the Plotly charts
            const retrievalChart = {retrieval_fig.to_json()};
            const generationBasicChart = {generation_basic_fig.to_json()};
            const generationQualityChart = {generation_quality_fig.to_json()};
            
            Plotly.newPlot('retrieval-chart', retrievalChart.data, retrievalChart.layout);
            Plotly.newPlot('generation-basic-chart', generationBasicChart.data, generationBasicChart.layout);
            Plotly.newPlot('generation-quality-chart', generationQualityChart.data, generationQualityChart.layout);
            
            // Also create copies for individual tabs
            Plotly.newPlot('retrieval-chart-tab', retrievalChart.data, retrievalChart.layout);
            Plotly.newPlot('generation-basic-chart-tab', generationBasicChart.data, generationBasicChart.layout);
            Plotly.newPlot('generation-quality-chart-tab', generationQualityChart.data, generationQualityChart.layout);
            
            // Tab functionality
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                }}
                
                tablinks = document.getElementsByClassName("tab");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                
                document.getElementById(tabName).className += " active";
                evt.currentTarget.className += " active";
                
                // Trigger resize to ensure charts render properly in tabs
                window.dispatchEvent(new Event('resize'));
            }}
        </script>
    </body>
    </html>
    """

    return HTMLString(html)


@step(enable_cache=False)
def visualize_evaluation_results(
    small_retrieval_eval_failure_rate: float,
    small_retrieval_eval_failure_rate_reranking: float,
    full_retrieval_eval_failure_rate: float,
    full_retrieval_eval_failure_rate_reranking: float,
    failure_rate_bad_answers: float,
    failure_rate_bad_immediate_responses: float,
    failure_rate_good_responses: float,
    average_toxicity_score: float,
    average_faithfulness_score: float,
    average_helpfulness_score: float,
    average_relevance_score: float,
) -> Annotated[HTMLString, "evaluation_dashboard"]:
    """
    Visualize the evaluation results by creating an interactive HTML dashboard.

    Args:
        small_retrieval_eval_failure_rate (float): Small retrieval evaluation failure rate.
        small_retrieval_eval_failure_rate_reranking (float): Small retrieval evaluation failure rate with reranking.
        full_retrieval_eval_failure_rate (float): Full retrieval evaluation failure rate.
        full_retrieval_eval_failure_rate_reranking (float): Full retrieval evaluation failure rate with reranking.
        failure_rate_bad_answers (float): Failure rate for bad answers.
        failure_rate_bad_immediate_responses (float): Failure rate for bad immediate responses.
        failure_rate_good_responses (float): Failure rate for good responses.
        average_toxicity_score (float): Average toxicity score.
        average_faithfulness_score (float): Average faithfulness score.
        average_helpfulness_score (float): Average helpfulness score.
        average_relevance_score (float): Average relevance score.

    Returns:
        str: HTML content for the interactive evaluation dashboard.
    """
    step_context = get_step_context()
    pipeline_run_name = step_context.pipeline_run.name

    # Calculate composite metrics
    composite_overall_quality = (
        average_faithfulness_score
        + average_helpfulness_score
        + average_relevance_score
    ) / 3

    composite_retrieval_effectiveness = (
        (1 - small_retrieval_eval_failure_rate / 100)
        + (1 - full_retrieval_eval_failure_rate / 100)
    ) / 2

    # Collect all metrics for dashboard and logging
    metrics_metadata = {
        # Retrieval metrics
        "retrieval.small_failure_rate": small_retrieval_eval_failure_rate,
        "retrieval.small_failure_rate_reranking": small_retrieval_eval_failure_rate_reranking,
        "retrieval.full_failure_rate": full_retrieval_eval_failure_rate,
        "retrieval.full_failure_rate_reranking": full_retrieval_eval_failure_rate_reranking,
        # Generation failure metrics
        "generation.failure_rate_bad_answers": failure_rate_bad_answers,
        "generation.failure_rate_bad_immediate": failure_rate_bad_immediate_responses,
        "generation.failure_rate_good": failure_rate_good_responses,
        # Quality metrics
        "quality.toxicity": average_toxicity_score,
        "quality.faithfulness": average_faithfulness_score,
        "quality.helpfulness": average_helpfulness_score,
        "quality.relevance": average_relevance_score,
        # Composite scores
        "composite.overall_quality": composite_overall_quality,
        "composite.retrieval_effectiveness": composite_retrieval_effectiveness,
    }

    # Log all metrics as metadata for dashboard visualization
    log_metadata(metadata=metrics_metadata)

    # Prepare data for visualization
    image1_labels = [
        "Small Retrieval Eval Failure Rate",
        "Small Retrieval Eval Failure Rate Reranking",
        "Full Retrieval Eval Failure Rate",
        "Full Retrieval Eval Failure Rate Reranking",
    ]
    # Note: No need to normalize scores for Plotly visualization
    image1_scores = [
        small_retrieval_eval_failure_rate,
        small_retrieval_eval_failure_rate_reranking,
        full_retrieval_eval_failure_rate,
        full_retrieval_eval_failure_rate_reranking,
    ]

    image2_labels = [
        "Failure Rate Bad Answers",
        "Failure Rate Bad Immediate Responses",
        "Failure Rate Good Responses",
    ]
    image2_scores = [
        failure_rate_bad_answers,
        failure_rate_bad_immediate_responses,
        failure_rate_good_responses,
    ]

    image3_labels = [
        "Average Toxicity Score",
        "Average Faithfulness Score",
        "Average Helpfulness Score",
        "Average Relevance Score",
    ]
    image3_scores = [
        average_toxicity_score,
        average_faithfulness_score,
        average_helpfulness_score,
        average_relevance_score,
    ]

    # Generate the HTML dashboard
    html_content = generate_evaluation_html(
        pipeline_run_name,
        image1_labels,
        image1_scores,
        image2_labels,
        image2_scores,
        image3_labels,
        image3_scores,
        metrics_metadata,
    )

    return html_content
