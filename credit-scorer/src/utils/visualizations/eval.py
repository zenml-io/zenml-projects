from typing import Any, Dict

import numpy as np
from sklearn.metrics import precision_recall_curve
from zenml.logger import get_logger
from zenml.types import HTMLString

logger = get_logger(__name__)


def _create_precision_recall_plot(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold_metrics: Dict[float, Dict[str, Any]],
    min_cost_threshold: float,
) -> str:
    """Create precision-recall curve plot and convert to base64.

    Args:
        y_test: The test set labels
        y_prob: The predicted probabilities
        threshold_metrics: Dictionary of metrics at different thresholds
        min_cost_threshold: The optimal threshold value

    Returns:
        str: Base64 encoded image string
    """
    import base64
    import io

    import matplotlib.pyplot as plt

    # Create precision-recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(recall_curve, precision_curve, marker=".", label="Model")

    # Mark points for different thresholds
    for threshold in threshold_metrics:
        metrics = threshold_metrics[threshold]
        plt.plot(
            metrics["recall"],
            metrics["precision"],
            "o",
            markersize=8,
            label=f"Threshold {threshold}",
        )

    # Highlight the optimal threshold point
    opt_metrics = threshold_metrics[min_cost_threshold]
    plt.plot(
        opt_metrics["recall"],
        opt_metrics["precision"],
        "o",
        markersize=12,
        color="red",
        label=f"Optimal (t={min_cost_threshold})",
    )

    plt.title("Precision-Recall Curve", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best")

    # Save figure to memory
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)

    # Convert to base64 for HTML embedding
    return base64.b64encode(buf.read()).decode("utf-8")


def _extract_metrics(
    performance_metrics: Dict[str, Any],
    threshold_metrics: Dict[float, Dict[str, Any]],
    min_cost_threshold: float,
) -> Dict[str, Any]:
    """Extract and organize metrics for the dashboard.

    Args:
        performance_metrics: Dictionary of performance metrics
        threshold_metrics: Dictionary of metrics at different thresholds
        min_cost_threshold: The optimal threshold value

    Returns:
        Dict: Organized metrics for the dashboard
    """
    # Extract key metrics
    metrics = {
        "accuracy": performance_metrics.get("accuracy", 0),
        "precision": performance_metrics.get("precision", 0),
        "recall": performance_metrics.get("recall", 0),
        "f1": performance_metrics.get("f1_score", 0),
        "auc": performance_metrics.get(
            "auc_roc", performance_metrics.get("auc", 0)
        ),
        "avg_precision": performance_metrics.get("average_precision", 0),
        # Confusion matrix components
        "tn": performance_metrics.get("true_negatives", 0),
        "fp": performance_metrics.get("false_positives", 0),
        "fn": performance_metrics.get("false_negatives", 0),
        "tp": performance_metrics.get("true_positives", 0),
    }

    # Get optimal metrics
    opt_metrics = threshold_metrics.get(min_cost_threshold, {})
    metrics.update(
        {
            "opt_precision": performance_metrics.get(
                "optimal_precision", opt_metrics.get("precision", 0)
            ),
            "opt_recall": performance_metrics.get(
                "optimal_recall", opt_metrics.get("recall", 0)
            ),
            "opt_f1": performance_metrics.get(
                "optimal_f1", opt_metrics.get("f1_score", 0)
            ),
            "opt_cost": performance_metrics.get(
                "optimal_cost", opt_metrics.get("normalized_cost", 0)
            ),
            "min_cost_threshold": min_cost_threshold,
        }
    )

    return metrics


def _build_html_dashboard(
    metrics: Dict[str, Any],
    threshold_metrics: Dict[float, Dict[str, Any]],
    precision_recall_img: str,
) -> str:
    """Build HTML dashboard with metrics and visualizations.

    Args:
        metrics: Extracted metrics dictionary
        threshold_metrics: Dictionary of metrics at different thresholds
        precision_recall_img: Base64 encoded image

    Returns:
        str: Complete HTML content
    """
    # CSS and JavaScript have been shortened for brevity
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2, h3 {{
                color: #1F4E79;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            .dashboard {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background-color: #f9f9f9;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                flex: 1;
                min-width: 200px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #1F4E79;
            }}
            .threshold-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .threshold-table th, .threshold-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }}
            .threshold-table th {{
                background-color: #f2f2f2;
            }}
            .optimal-row {{
                background-color: #e6f4ea;
                font-weight: bold;
            }}
            .tabs {{
                margin-top: 30px;
            }}
            .tab-buttons {{
                display: flex;
                gap: 5px;
            }}
            .tab-button {{
                padding: 10px 20px;
                border: none;
                background-color: #f2f2f2;
                cursor: pointer;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }}
            .tab-button.active {{
                background-color: #1F4E79;
                color: white;
            }}
            .tab-content {{
                display: none;
                border: 1px solid #ddd;
                padding: 20px;
            }}
            .tab-content.active {{
                display: block;
            }}
            .chart-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-radius: 5px;
            }}
        </style>
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tabbuttons;
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tabbuttons = document.getElementsByClassName("tab-button");
                for (i = 0; i < tabbuttons.length; i++) {{
                    tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }}
            
            window.onload = function() {{
                document.getElementById("defaultOpen").click();
            }};
        </script>
    </head>
    <body>
        <h1>Credit Scoring Model Evaluation Dashboard</h1>
        
        <h2>Performance Overview</h2>
        <div class="dashboard">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="metric-value">{metrics["accuracy"]:.2%}</div>
                <p>Overall classification accuracy</p>
            </div>
            <div class="metric-card">
                <h3>AUC-ROC</h3>
                <div class="metric-value">{metrics["auc"]:.2%}</div>
                <p>Area under ROC curve</p>
            </div>
            <div class="metric-card">
                <h3>F1 Score</h3>
                <div class="metric-value">{metrics["f1"]:.2%}</div>
                <p>Harmonic mean of precision and recall</p>
            </div>
            <div class="metric-card">
                <h3>Optimal F1</h3>
                <div class="metric-value">{metrics["opt_f1"]:.2%}</div>
                <p>At threshold {metrics["min_cost_threshold"]}</p>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab-buttons">
                <button class="tab-button active" onclick="openTab(event, 'threshold-tab')" id="defaultOpen">Threshold Analysis</button>
                <button class="tab-button" onclick="openTab(event, 'metrics-tab')">Detailed Metrics</button>
                <button class="tab-button" onclick="openTab(event, 'confusion-tab')">Confusion Matrix</button>
            </div>
            
            <div id="threshold-tab" class="tab-content active">
                <h3>Precision-Recall Curve</h3>
                <div class="chart-container">
                    <img src="data:image/png;base64,{precision_recall_img}" alt="Precision-Recall Curve">
                </div>
                
                <h3>Threshold Comparison</h3>
                <p>Different threshold values and their impact on model performance metrics:</p>
                <table class="threshold-table">
                    <tr>
                        <th>Threshold</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Cost</th>
                    </tr>
    """

    # Add rows for each threshold
    for threshold in sorted(threshold_metrics.keys()):
        th_metrics = threshold_metrics[threshold]
        row_class = (
            "optimal-row" if threshold == metrics["min_cost_threshold"] else ""
        )
        html_content += f"""
                    <tr class="{row_class}">
                        <td>{threshold}</td>
                        <td>{th_metrics.get("precision", 0):.4f}</td>
                        <td>{th_metrics.get("recall", 0):.4f}</td>
                        <td>{th_metrics.get("f1_score", 0):.4f}</td>
                        <td>{th_metrics.get("normalized_cost", 0):.4f}</td>
                    </tr>
        """

    html_content += """
                </table>
            </div>
            
            <div id="metrics-tab" class="tab-content">
                <h3>Standard Metrics (at threshold 0.5)</h3>
                <table class="threshold-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
    """

    # Add standard metrics
    metrics_descriptions = [
        (
            "Accuracy",
            f"{metrics['accuracy']:.4f}",
            "Proportion of correctly classified instances",
        ),
        (
            "Precision",
            f"{metrics['precision']:.4f}",
            "True positives / (True positives + False positives)",
        ),
        (
            "Recall",
            f"{metrics['recall']:.4f}",
            "True positives / (True positives + False negatives)",
        ),
        (
            "F1 Score",
            f"{metrics['f1']:.4f}",
            "Harmonic mean of precision and recall",
        ),
        (
            "AUC-ROC",
            f"{metrics['auc']:.4f}",
            "Area under the Receiver Operating Characteristic curve",
        ),
        (
            "Average Precision",
            f"{metrics['avg_precision']:.4f}",
            "Average precision score across all recall levels",
        ),
    ]

    for metric, value, description in metrics_descriptions:
        html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{value}</td>
                        <td>{description}</td>
                    </tr>
        """

    html_content += """
                </table>
                
                <h3>Optimal Metrics</h3>
                <table class="threshold-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Threshold</th>
                    </tr>
    """

    # Add optimal metrics
    optimal_metrics = [
        (
            "Optimal Precision",
            f"{metrics['opt_precision']:.4f}",
            f"{metrics['min_cost_threshold']}",
        ),
        (
            "Optimal Recall",
            f"{metrics['opt_recall']:.4f}",
            f"{metrics['min_cost_threshold']}",
        ),
        (
            "Optimal F1 Score",
            f"{metrics['opt_f1']:.4f}",
            f"{metrics['min_cost_threshold']}",
        ),
        (
            "Optimal Cost",
            f"{metrics['opt_cost']:.4f}",
            f"{metrics['min_cost_threshold']}",
        ),
    ]

    for metric, value, threshold in optimal_metrics:
        html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{value}</td>
                        <td>{threshold}</td>
                    </tr>
        """

    # Add confusion matrix and insights
    html_content += f"""
                </table>
            </div>
            
            <div id="confusion-tab" class="tab-content">
                <h3>Confusion Matrix</h3>
                <p>Visual representation of model predictions vs actual values:</p>
                <div style="margin: 0 auto; max-width: 600px;">
                    <table class="threshold-table">
                        <tr>
                            <th></th>
                            <th>Predicted: No Default (0)</th>
                            <th>Predicted: Default (1)</th>
                        </tr>
                        <tr>
                            <th>Actual: No Default (0)</th>
                            <td style="background-color:#c6efce;">{metrics["tn"]} (True Negatives)</td>
                            <td style="background-color:#ffc7ce;">{metrics["fp"]} (False Positives)</td>
                        </tr>
                        <tr>
                            <th>Actual: Default (1)</th>
                            <td style="background-color:#ffc7ce;">{metrics["fn"]} (False Negatives)</td>
                            <td style="background-color:#c6efce;">{metrics["tp"]} (True Positives)</td>
                        </tr>
                    </table>
                </div>
                
                <h3>Interpretation</h3>
                <ul>
                    <li><strong>True Negatives ({metrics["tn"]}):</strong> Correctly identified non-defaults</li>
                    <li><strong>False Positives ({metrics["fp"]}):</strong> Incorrectly flagged as defaults</li>
                    <li><strong>False Negatives ({metrics["fn"]}):</strong> Defaults missed by the model</li>
                    <li><strong>True Positives ({metrics["tp"]}):</strong> Correctly identified defaults</li>
                </ul>
                
                <p><strong>Note:</strong> In credit scoring, False Negatives (missed defaults) are typically more costly than False Positives (wrongly declined creditworthy customers).</p>
            </div>
        </div>
        
        <h2>Insights and Recommendations</h2>
        <ul>
            <li>The model achieves an AUC of {metrics["auc"]:.2%}, indicating good discriminative ability.</li>
            <li>The optimal threshold for minimizing cost is {metrics["min_cost_threshold"]}, yielding a cost of {metrics["opt_cost"]:.4f}.</li>
            <li>At this threshold, precision is {metrics["opt_precision"]:.2%} and recall is {metrics["opt_recall"]:.2%}.</li>
            <li>The model correctly identifies {metrics["opt_recall"]:.2%} of actual defaults (Recall) while maintaining {metrics["opt_precision"]:.2%} precision.</li>
        </ul>
    </body>
    </html>
    """

    return html_content


def generate_eval_visualization(
    performance_metrics: Dict[str, Any],
    threshold_metrics: Dict[float, Dict[str, Any]],
    min_cost_threshold: float,
    y_test: np.ndarray,
    y_prob: np.ndarray,
) -> HTMLString:
    """Generate an HTML visualization for model evaluation metrics.

    Args:
        performance_metrics: Dictionary of performance metrics
        threshold_metrics: Dictionary of metrics at different thresholds
        min_cost_threshold: The optimal threshold value
        y_test: The test set labels
        y_prob: The predicted probabilities

    Returns:
        HTMLString: HTML visualization of model performance
    """
    try:
        # Extract relevant metrics
        metrics = _extract_metrics(
            performance_metrics, threshold_metrics, min_cost_threshold
        )

        # Create precision-recall curve plot
        precision_recall_img = _create_precision_recall_plot(
            y_test, y_prob, threshold_metrics, min_cost_threshold
        )

        # Build HTML dashboard
        html_content = _build_html_dashboard(
            metrics, threshold_metrics, precision_recall_img
        )

        return HTMLString(html_content)
    except Exception as e:
        logger.warning(f"Could not generate visualization: {e}")
        return HTMLString("<h1>Visualization generation failed</h1>")
