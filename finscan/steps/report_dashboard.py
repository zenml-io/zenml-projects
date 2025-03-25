import re
import json
from typing import Dict, Any
from zenml import step
from zenml.logger import get_logger
from zenml.types import HTMLString
from utils import extract_metrics, generate_context_html, generate_metrics_html

logger = get_logger(__name__)


@step
def financial_dashboard(response: Dict[str, Any]) -> HTMLString:
    """
    Generates an HTML dashboard artifact visualizing extracted financial insights.

    Args:
        response: The structured financial report response.

    Returns:
        A ZenML VisualizationArtifact pointing to the generated HTML file.
    """
    print(response)
    extracted_data = extract_metrics(response)

    # HTML template
    html_content = f"""
    <html>
    <head>
        <title>Financial Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; padding: 20px; background-color: #f4f4f4; }}
            .container {{ max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 5px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #007BFF; }}
            p, li {{ font-size: 14px; line-height: 1.6; color: #555; }}
            ul {{ padding-left: 20px; }}
            .metric-box {{ background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Financial Analysis Dashboard</h1>

            <h2>📊 Key Financial Metrics</h2>
            {generate_metrics_html(extracted_data["metrics"])}

            <h2>📰 Market Context</h2>
            {generate_context_html(extracted_data["context"])}

            <h2>🏆 Competitor Insights</h2>
            <p>{extracted_data["competitor"]}</p>

            <h2>⚠️ Contradictions & Gap Analysis</h2>
            <p>{extracted_data["contradictions"]}</p>

            <h2>📝 Additional Context</h2>
            <p>{extracted_data["additional_context"]}</p>
        </div>
    </body>
    </html>
    """

    logger.info("Generated Financial Dashboard HTML.")

    return HTMLString(html_content)