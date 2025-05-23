"""Compliance dashboard HTML rendering components.

This module provides functions to generate HTML for the compliance dashboard,
extracted from streamlit_app for reusability between the streamlit app and
ZenML artifacts.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from zenml.types import HTMLString

logger = logging.getLogger(__name__)

# CSS styling for the dashboard
DASHBOARD_CSS = """
<style>
    body {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f8f9fa;
    }
    h1, h2, h3, h4 {
        color: #1F4E79;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    h1 {
        border-bottom: 2px solid #1F4E79;
        padding-bottom: 10px;
    }
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .summary-header {
        text-align: center;
        padding: 10px;
        background-color: #f0f5ff;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .overall-score {
        font-size: 32px;
        font-weight: bold;
        color: #1F4E79;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric-card {
        flex: 1;
        min-width: 200px;
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .strongest {
        border-left: 4px solid #478C5C;
    }
    .strongest .metric-value {
        color: #478C5C;
    }
    .weakest {
        border-left: 4px solid #D64045;
    }
    .weakest .metric-value {
        color: #D64045;
    }
    .articles-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 15px;
        margin-bottom: 25px;
    }
    .article-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        position: relative;
    }
    .article-title {
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 14px;
        color: #1F4E79;
    }
    .article-value {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .progress-bar {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 5px;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
    }
    .progress-high {
        background-color: #C6E0B4;
    }
    .progress-medium {
        background-color: #FFE699;
    }
    .progress-low {
        background-color: #F8CBAD;
    }
    .article-source {
        font-size: 11px;
        color: #666;
    }
    .gauges-container {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        justify-content: space-between;
    }
    .gauge-card {
        flex: 1;
        min-width: 300px;
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .gauge-title {
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    .gauge {
        width: 100%;
        height: 120px;
        position: relative;
        margin-bottom: 10px;
    }
    .data-sources {
        text-align: center;
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }
    .findings-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .finding {
        padding: 12px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .finding-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .finding-critical {
        background-color: #ffecec;
        border-left: 3px solid #D64045;
    }
    .finding-warning {
        background-color: #fff9ec;
        border-left: 3px solid #FFB30F;
    }
    .finding-positive {
        background-color: #f0f7f0;
        border-left: 3px solid #478C5C;
    }
    .metrics-section {
        margin-top: 30px;
    }
    .metrics-header {
        padding: 10px;
        background-color: #f0f5ff;
        border-radius: 5px;
        margin-bottom: 15px;
        text-align: center;
    }
    .findings-section {
        margin-top: 30px;
    }
    .footer-bar {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .footer-progress {
        background-color: #e0e0e0;
        border-radius: 5px;
        height: 10px;
        width: 100%;
        margin: 5px 0;
    }
    .footer-progress-fill {
        background-color: #478C5C;
        height: 100%;
        border-radius: 5px;
    }
    /* Improved gauge styling */
    .gauge-container {
        width: 100%;
        height: 180px;
        position: relative;
        margin-bottom: 10px;
    }
    .gauge-background {
        width: 100%;
        height: 15px;
        background-color: #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 1s ease-in-out;
    }
    .gauge-markers {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
        font-size: 12px;
        color: #666;
    }
    .gauge-value {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    .gauge-threshold {
        position: absolute;
        top: 0;
        height: 15px;
        width: 3px;
        background-color: black;
        z-index: 10;
    }
</style>
"""

# Primary colors
PRIMARY_COLOR = "#1F4E79"
RISK_COLORS = {
    "HIGH": "#D64045",
    "MEDIUM": "#FFB30F",
    "LOW": "#478C5C",
}


def format_activities(
    activities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Format activity dates (including ISO/T and microseconds) for display.

    Args:
        activities: List of activity dictionaries containing date information

    Returns:
        List of activities with formatted dates added
    """
    for activity in activities:
        date_val = activity.get("date")
        if isinstance(date_val, str):
            try:
                # try ISO‐format parser (handles T separator and microseconds)
                dt = datetime.fromisoformat(date_val)
            except ValueError:
                try:
                    dt = datetime.strptime(date_val, "%Y-%m-%d")
                except ValueError:
                    # give up and keep the original
                    activity["formatted_date"] = date_val
                    continue

            if dt.hour or dt.minute or dt.second:
                activity["formatted_date"] = dt.strftime("%b %d, %Y at %H:%M")
            else:
                activity["formatted_date"] = dt.strftime("%b %d, %Y")
        else:
            # non‐string dates (datetime already, etc.)
            try:
                activity["formatted_date"] = (
                    date_val.strftime("%b %d, %Y at %H:%M")
                    if hasattr(date_val, "hour")
                    else str(date_val)
                )
            except Exception:
                activity["formatted_date"] = str(date_val)

    return activities


def render_article_card(
    article: str, score: float, sources_text: str = "N/A"
) -> str:
    """Render an article card for compliance visualization.

    Args:
        article: The article title
        score: The compliance score (0-100)
        sources_text: Text describing data sources

    Returns:
        HTML string for the article card
    """
    # Determine color based on score
    if score >= 80:
        color = "#478C5C"  # green
        progress_class = "progress-high"
    elif score >= 60:
        color = "#FFB30F"  # yellow
        progress_class = "progress-medium"
    else:
        color = "#D64045"  # red
        progress_class = "progress-low"

    # Create article card HTML
    card_html = f"""
    <div class="article-card">
        <div class="article-title">{article}</div>
        <div class="article-value" style="color: {color};">{score:.1f}%</div>
        <div class="progress-bar">
            <div class="progress-fill {progress_class}" style="width: {score}%;"></div>
        </div>
        <div class="article-source"><strong>Sources:</strong> {sources_text}</div>
    </div>
    """
    return card_html


def render_gauge(
    article: str,
    score: float,
    sources_text: str = "N/A",
    threshold: float = 75,
) -> str:
    """Render a gauge for article compliance visualization.

    Args:
        article: The article title
        score: The compliance score (0-100)
        sources_text: Text describing data sources
        threshold: The threshold value for the gauge (0-100)

    Returns:
        HTML string for the gauge
    """
    # Determine color based on score
    if score >= 80:
        color = "#C6E0B4"  # green
    elif score >= 60:
        color = "#FFE699"  # yellow
    else:
        color = "#ffcccb"  # red

    # Create improved gauge HTML
    gauge_html = f"""
    <div class="gauge-card">
        <div class="gauge-title">{article}</div>
        <div class="gauge-value">{score:.1f}%</div>
        <div class="gauge-container">
            <div class="gauge-background">
                <div class="gauge-fill" style="width: {score}%; background-color: {color};"></div>
                <div class="gauge-threshold" style="left: {threshold}%;"></div>
            </div>
            <div class="gauge-markers">
                <div>0%</div>
                <div>25%</div>
                <div>50%</div>
                <div>75%</div>
                <div>100%</div>
            </div>
        </div>
        <div class="data-sources">
            <strong>Sources:</strong> {sources_text}
        </div>
    </div>
    """
    return gauge_html


def render_finding(finding: Dict[str, Any]) -> str:
    """Render a single compliance finding.

    Args:
        finding: Dictionary containing finding information

    Returns:
        HTML string for the finding
    """
    finding_type = finding.get("type", "warning").lower()
    title = finding.get("title", finding.get("message", "Finding"))
    description = finding.get("description", finding.get("recommendation", ""))

    css_class = f"finding-{finding_type}"

    html = f"""
    <div class="finding {css_class}">
        <div class="finding-title">{title}</div>
        <div class="finding-description">{description}</div>
    </div>
    """
    return html


def generate_compliance_dashboard_html(
    compliance_results: Dict[str, Any],
    risk_df: Optional[Any] = None,
    incident_df: Optional[Any] = None,
) -> str:
    """Generate HTML for the compliance dashboard.

    Args:
        compliance_results: Dictionary with compliance calculation results
        risk_df: Optional DataFrame with risk data
        incident_df: Optional DataFrame with incident data

    Returns:
        Dashboard HTML as a string
    """
    # Start with HTML head and CSS
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EU AI Act Compliance Dashboard</title>
        {DASHBOARD_CSS}
    </head>
    <body>
    """

    # Add dashboard title
    html += """
    <h1>EU AI Act Compliance Dashboard</h1>
    <div class="card">
        <h2>Executive Summary</h2>
        <p>Overview of system compliance status, risk indicators, and key metrics related to the EU AI Act requirements.</p>
    </div>
    """

    # Process compliance summary
    from streamlit_app.data.compliance_utils import get_compliance_summary

    compliance_summary = get_compliance_summary(compliance_results)

    # Extract risk metrics
    total_risks = 0
    high_risks = 0
    completion_status = 0

    if risk_df is not None:
        try:
            severity_column = next(
                (
                    col
                    for col in ["risk_category", "risk_level"]
                    if col in risk_df.columns
                ),
                None,
            )

            if severity_column:
                total_risks = len(risk_df)
                high_risks = (risk_df[severity_column] == "HIGH").sum()
                completion_status = (
                    (risk_df.get("status", "") == "COMPLETED").sum()
                    / total_risks
                    if total_risks > 0
                    else 0
                )
        except Exception as e:
            logger.warning(f"Error processing risk data: {e}")

    # Calculate compliance metrics
    critical_findings = compliance_summary.get("critical_count", 0)
    warning_findings = compliance_summary.get("warning_count", 0)
    strongest_article = compliance_summary.get(
        "strongest_article", {"name": "None", "score": 0}
    )
    weakest_article = compliance_summary.get(
        "weakest_article", {"name": "None", "score": 0}
    )

    # Format article names for display
    strongest_name = strongest_article["name"]
    if "Art." in strongest_name and "(" in strongest_name:
        strongest_name = strongest_name.split("(")[0].strip()

    weakest_name = weakest_article["name"]
    if "Art." in weakest_name and "(" in weakest_name:
        weakest_name = weakest_name.split("(")[0].strip()

    # Add overall compliance score
    overall_score = compliance_summary.get("overall_score", 0)

    html += f"""
    <div class="summary-header">
        <h3 style="margin: 0; color:#1F4E79;">Overall Compliance Score</h3>
        <div class="overall-score">{overall_score:.1f}%</div>
    </div>
    """

    # Get detailed compliance information
    from streamlit_app.data.compliance_utils import (
        format_compliance_findings,
        get_compliance_data_sources,
    )
    from streamlit_app.data.processor import compute_article_compliance

    article_compliance = compute_article_compliance(
        risk_df, use_compliance_calculator=True
    )
    data_sources = get_compliance_data_sources(compliance_results)
    findings = compliance_results.get("findings", [])
    grouped_findings = format_compliance_findings(findings)

    # Add Article Cards Grid (NEW LAYOUT)
    html += "<h3>Article Compliance Status</h3>"
    html += '<div class="articles-grid">'

    # Create article cards for each article (excluding Overall Compliance)
    article_items = [
        (k, v)
        for k, v in article_compliance.items()
        if k != "Overall Compliance"
    ]

    for article, score in article_items:
        # Extract article ID for looking up data sources
        article_id = None
        if article.startswith("Art."):
            article_num = article.split("(")[0].strip().replace("Art. ", "")
            article_id = f"article_{article_num}"

        # Get data sources for this article
        sources = data_sources.get(article_id, [])
        sources_text = ", ".join(sources) if sources else "N/A"

        # Add article card
        html += render_article_card(article, score, sources_text)

    html += "</div>"  # Close articles-grid

    # Risk Metrics Section (MOVED BELOW)
    html += """
    <div class="metrics-section">
        <div class="metrics-header">
            <h3 style="margin: 0; color:#1F4E79;">Risk & Compliance Metrics</h3>
        </div>
    """

    # Add metrics container
    html += """
        <div class="metrics-container">
    """

    html += f"""
            <div class="metric-card">
                <div class="metric-value">{total_risks}</div>
                <div class="metric-label">Risks Identified</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{critical_findings}</div>
                <div class="metric-label">Critical Findings</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{warning_findings}</div>
                <div class="metric-label">Warning Findings</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{completion_status:.0%}</div>
                <div class="metric-label">Mitigation Progress</div>
            </div>
        
            <div class="metric-card">
                <div class="metric-value">{high_risks}</div>
                <div class="metric-label">High Risk Items</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{incident_df.shape[0] if incident_df is not None else 0}</div>
                <div class="metric-label">Reported Incidents</div>
            </div>
            
            <div class="metric-card strongest">
                <div class="metric-value">{strongest_article["score"]:.0f}%</div>
                <div class="metric-label">Strongest: {strongest_name}</div>
            </div>
            
            <div class="metric-card weakest">
                <div class="metric-value">{weakest_article["score"]:.0f}%</div>
                <div class="metric-label">Weakest: {weakest_name}</div>
            </div>
    """

    html += """
        </div>
    </div>
    """  # Close metrics container and section

    # Findings Section (MOVED BELOW)
    html += """
    <div class="findings-section">
        <div class="card">
            <h3>Key Compliance Findings</h3>
            <div class="findings-container">
    """

    # Critical issues first
    if grouped_findings["critical"]:
        html += '<div style="color:#D64045; font-weight:bold; margin-bottom: 5px;">Critical Issues:</div>'
        for finding in grouped_findings["critical"][:3]:  # Show top 3
            html += render_finding(finding)

    # Warnings next
    if grouped_findings["warning"]:
        html += '<div style="color:#FFB30F; font-weight:bold; margin-top: 15px; margin-bottom: 5px;">Warnings:</div>'
        for finding in grouped_findings["warning"][:3]:  # Show top 3
            html += render_finding(finding)

    # Positive findings
    if grouped_findings["positive"]:
        html += '<div style="color:#478C5C; font-weight:bold; margin-top: 15px; margin-bottom: 5px;">Positive Findings:</div>'
        for finding in grouped_findings["positive"][:3]:  # Show top 3
            html += render_finding(finding)

    # If no findings, show a message
    if not any(grouped_findings.values()):
        html += '<div style="padding: 15px; background-color: #e6f7ff; border-radius: 5px;">No compliance findings available for this release.</div>'

    html += """
            </div>
        </div>
    </div>
    """  # Close findings section

    # Add compliance status bar
    compliance_percentage = compliance_summary.get("overall_score", 0)
    last_release_id = compliance_summary.get("release_id", "Unknown")

    # Determine color based on compliance score
    bar_color = (
        "#D64045"
        if compliance_percentage < 60
        else "#FFB30F"
        if compliance_percentage < 80
        else "#478C5C"
    )

    # Determine status text
    if compliance_percentage >= 80:
        status_text = "High Compliance"
    elif compliance_percentage >= 60:
        status_text = "Moderate Compliance"
    else:
        status_text = "Low Compliance"

    # Create status bar (using updated footer-bar classes)
    html += f"""
    <div class="footer-bar">
        <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <div><strong>EU AI Act Compliance Status:</strong> {status_text}</div>
            <div style="color:#666; font-size:12px;">Release ID: {last_release_id}</div>
        </div>
        <div class="footer-progress">
            <div class="footer-progress-fill" style="width:{compliance_percentage}%;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:5px;">
            <div>0%</div>
            <div>50%</div>
            <div>100%</div>
        </div>
    </div>
    """

    # Add generated timestamp
    html += f"""
    <div style="text-align: center; font-size: 12px; color: #666; margin-top: 30px; padding-top: 10px; border-top: 1px solid #eee;">
        Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}
    </div>
    """

    # Close HTML tags
    html += """
    </body>
    </html>
    """

    return html


def create_compliance_dashboard_artifact(
    compliance_results: Dict[str, Any],
    risk_df: Optional[Any] = None,
    incident_df: Optional[Any] = None,
) -> HTMLString:
    """Create a ZenML HTML artifact for the compliance dashboard.

    Args:
        compliance_results: Dictionary with compliance calculation results
        risk_df: Optional DataFrame with risk data
        incident_df: Optional DataFrame with incident data

    Returns:
        ZenML HTMLString artifact
    """
    html_content = generate_compliance_dashboard_html(
        compliance_results=compliance_results,
        risk_df=risk_df,
        incident_df=incident_df,
    )

    return HTMLString(html_content)
