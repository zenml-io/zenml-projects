from typing import Dict, Any
import ast
import re

def extract_metrics(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts financial metrics and key insights from the structured response.

    Args:
        response: A structured dictionary containing task outcomes and context.

    Returns:
        A dictionary containing extracted financial metrics and relevant insights.
    """
    extracted_data = {
        "metrics": {},
        "context": {},
        "competitor": "",
        "contradictions": "",
        "additional_context": ""
    }

    task_outcome = response.get("### 2. Task outcome (extremely detailed version)", {})

    # Extract financial metrics
    metric_results = task_outcome.get("Metric Results", "")
    extracted_data["metrics"] = parse_financial_metrics(metric_results)

    # Extract context results
    extracted_data["context"] = task_outcome.get("Context Result", {})

    # Extract competitor information
    extracted_data["competitor"] = task_outcome.get("Competitor Result", "")

    # Extract contradictions and gap analysis
    extracted_data["contradictions"] = task_outcome.get("Contradictory and Gap Analysis", "")

    # Extract additional context
    extracted_data["additional_context"] = response.get("### 3. Additional context (if relevant)", "")

    return extracted_data

def parse_financial_metrics(text: str) -> Dict[str, str]:
    """
    Parses financial metric values from the metric result text.

    Args:
        text: The text containing financial metric information.

    Returns:
        A dictionary with extracted financial metrics.
    """
    metrics = {}

    # Define patterns for extracting key financial metrics
    patterns = {
        "Revenue Growth Rate": r"Revenue Growth Rate: ([\d\.%]+)",
        "Net Margin": r"Net Margin: ([\d\.%]+)",
        "Debt-to-Equity Ratio": r"Debt-to-Equity Ratio: ([\d\.%]+)"
    }

    # Search for matches in the provided text
    for metric, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[metric] = match.group(1)

    return metrics


def generate_metrics_html(metrics: Dict[str, str]) -> str:
    """
    Generates HTML for financial metrics.

    Args:
        metrics: Dictionary of extracted financial metrics.

    Returns:
        A string containing HTML content.
    """
    if not metrics:
        return "<p>No financial metrics available.</p>"

    html = "<ul>"
    for metric, value in metrics.items():
        html += f'<li class="metric-box"><strong>{metric}:</strong> {value}</li>'
    html += "</ul>"
    return html

def string_to_dict(input_str):
    try:
        input_str = input_str.replace("Here is the final answer from your managed agent 'None':", "").strip()
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error converting string to dict: {e}")
        return None

def generate_context_html(context: Dict[str, Any]) -> str:
    """
    Generates HTML for market context insights.

    Args:
        context: Dictionary containing market-related context.

    Returns:
        A string containing HTML content.
    """
    if not context:
        return "<p>No market context available.</p>"

    html = "<ul>"
    for key, value in context.items():
        html += f"<li><strong>{key}:</strong> {value}</li>"
    html += "</ul>"
    return html
