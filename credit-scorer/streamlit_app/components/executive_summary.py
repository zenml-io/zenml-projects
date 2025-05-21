"""Executive Summary component for the dashboard."""

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
from src.utils.visualizations.compliance_dashboard import (
    format_activities,
)
from streamlit.components.v1 import html

from streamlit_app.config import PRIMARY_COLOR, RISK_COLORS
from streamlit_app.data.compliance_utils import (
    format_compliance_findings,
    get_compliance_data_sources,
    get_compliance_results,
    get_compliance_summary,
    get_last_update_timestamps,
)
from streamlit_app.data.processor import compute_article_compliance

# We are now importing format_activities from the shared module
# So we no longer need this duplicate function here


def display_exec_summary(risk_df, incident_df):
    """Display an executive summary dashboard."""
    st.markdown(
        '<div class="card">'
        "<h2>Executive Summary</h2>"
        "<p>Overview of system compliance status, risk indicators, and key metrics related to the EU AI Act requirements.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Key metrics - combine compliance results with risk data
    col1, col2, col3, col4 = st.columns(4)

    # Get compliance data
    compliance_results = get_compliance_results()
    compliance_summary = get_compliance_summary(compliance_results)

    # Extract risk metrics
    if risk_df is not None:
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
                (risk_df.get("status", "") == "COMPLETED").sum() / total_risks
                if total_risks > 0
                else 0
            )
        else:
            total_risks = 0
            high_risks = 0
            completion_status = 0
    else:
        total_risks = 0
        high_risks = 0
        completion_status = 0

    # Incident metrics
    total_incidents = len(incident_df) if incident_df is not None else 0

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

    # Display metrics in styled cards - Top Row
    with col1:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="metric-value">{total_risks}</div>'
            '<div class="metric-label">Risks Identified</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="metric-value">{critical_findings}</div>'
            '<div class="metric-label">Critical Findings</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="metric-value">{warning_findings}</div>'
            '<div class="metric-label">Warning Findings</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="metric-value">{completion_status:.0%}</div>'
            '<div class="metric-label">Mitigation Progress</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="metric-value">{high_risks}</div>'
            '<div class="metric-label">High Risk Items</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="metric-card">'
            f'<div class="metric-value">{total_incidents}</div>'
            '<div class="metric-label">Reported Incidents</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col3:
        # Show strongest article
        st.markdown(
            f'<div class="metric-card" style="border-left: 4px solid #478C5C;">'
            f'<div class="metric-value" style="color: #478C5C;">{strongest_article["score"]:.0f}%</div>'
            f'<div class="metric-label">Strongest: {strongest_name}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col4:
        # Show weakest article
        st.markdown(
            f'<div class="metric-card" style="border-left: 4px solid #D64045;">'
            f'<div class="metric-value" style="color: #D64045;">{weakest_article["score"]:.0f}%</div>'
            f'<div class="metric-label">Weakest: {weakest_name}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    # Compliance status by EU AI Act article
    st.markdown("<h3>EU AI Act Compliance Status</h3>", unsafe_allow_html=True)

    # Get detailed compliance information
    compliance_results = get_compliance_results()
    article_compliance = compute_article_compliance(
        risk_df, use_compliance_calculator=True
    )
    data_sources = get_compliance_data_sources(compliance_results)
    findings = compliance_results.get("findings", [])
    grouped_findings = format_compliance_findings(findings)
    update_timestamps = get_last_update_timestamps(compliance_results)

    # Display overall compliance score in a prominent way
    if "Overall Compliance" in article_compliance:
        overall_score = article_compliance["Overall Compliance"]
        st.markdown(
            f"<div style='text-align:center; padding:10px; background-color:#f0f5ff; border-radius:5px; margin-bottom:15px;'>"
            f"<h4 style='margin:0; color:#1F4E79;'>Overall Compliance Score</h4>"
            f"<div style='font-size:32px; font-weight:bold; color:#1F4E79;'>{overall_score:.1f}%</div>"
            f"<div style='font-size:12px; color:#666;'>Last updated: {update_timestamps.get('Model Evaluation', 'N/A')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Create a two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create a gauge chart for each article (excluding Overall Compliance)
        gauge_cols = st.columns(3)
        article_items = [
            (k, v)
            for k, v in article_compliance.items()
            if k != "Overall Compliance"
        ]

        for i, (article, score) in enumerate(article_items):
            with gauge_cols[i % 3]:
                # Extract article ID for looking up data sources
                article_id = None
                if article.startswith("Art."):
                    article_num = (
                        article.split("(")[0].strip().replace("Art. ", "")
                    )
                    article_id = f"article_{article_num}"

                # Get data sources for this article
                sources = data_sources.get(article_id, [])
                sources_text = ", ".join(sources) if sources else "N/A"

                # Use the original plotly gauge for interactive features
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": PRIMARY_COLOR},
                            "steps": [
                                {"range": [0, 60], "color": "#ffcccb"},
                                {"range": [60, 80], "color": "#FFE699"},
                                {"range": [80, 100], "color": "#C6E0B4"},
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": 75,
                            },
                        },
                        title={"text": article, "font": {"size": 14}},
                        number={"suffix": "%", "font": {"size": 20}},
                    )
                )
                fig.update_layout(
                    height=180, margin=dict(l=30, r=30, t=50, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show data sources under each gauge
                st.markdown(
                    f"<div style='text-align:center; font-size:11px; color:#666; margin-top:-15px;'>"
                    f"<strong>Sources:</strong> {sources_text}</div>",
                    unsafe_allow_html=True,
                )

    with col2:
        # Show compliance findings
        st.markdown("<h4>Key Compliance Findings</h4>", unsafe_allow_html=True)

        # Critical issues first
        if grouped_findings["critical"]:
            st.markdown(
                "<span style='color:#D64045; font-weight:bold;'>Critical Issues:</span>",
                unsafe_allow_html=True,
            )
            for finding in grouped_findings["critical"][:3]:  # Show top 3
                st.markdown(
                    f"<div style='padding:8px; margin-bottom:5px; background-color:#ffecec; border-left:3px solid #D64045; border-radius:3px;'>"
                    f"<div style='font-weight:bold;'>{finding.get('title', 'Issue')}</div>"
                    f"<div style='font-size:12px;'>{finding.get('description', '')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Warnings next
        if grouped_findings["warning"]:
            st.markdown(
                "<span style='color:#FFB30F; font-weight:bold;'>Warnings:</span>",
                unsafe_allow_html=True,
            )
            for finding in grouped_findings["warning"][:3]:  # Show top 3
                st.markdown(
                    f"<div style='padding:8px; margin-bottom:5px; background-color:#fff9ec; border-left:3px solid #FFB30F; border-radius:3px;'>"
                    f"<div style='font-weight:bold;'>{finding.get('title', 'Warning')}</div>"
                    f"<div style='font-size:12px;'>{finding.get('description', '')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Positive findings
        if grouped_findings["positive"]:
            st.markdown(
                "<span style='color:#478C5C; font-weight:bold;'>Positive Findings:</span>",
                unsafe_allow_html=True,
            )
            for finding in grouped_findings["positive"][:3]:  # Show top 3
                st.markdown(
                    f"<div style='padding:8px; margin-bottom:5px; background-color:#f0f7f0; border-left:3px solid #478C5C; border-radius:3px;'>"
                    f"<div style='font-weight:bold;'>{finding.get('title', 'Positive')}</div>"
                    f"<div style='font-size:12px;'>{finding.get('description', '')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # If no findings, show a message
        if not any(grouped_findings.values()):
            st.info("No compliance findings available for this release.")

    # Recent activities section
    st.markdown("<h3>Recent Activities</h3>", unsafe_allow_html=True)

    # Show compliance status bar
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

    # Create status bar
    st.markdown(
        f"""
        <div style="background-color:#f5f5f5; padding:10px; border-radius:5px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <div><strong>EU AI Act Compliance Status:</strong> {status_text}</div>
                <div style="color:#666; font-size:12px;">Release ID: {last_release_id}</div>
            </div>
            <div style="background-color:#e0e0e0; border-radius:5px; height:10px; width:100%;">
                <div style="background-color:{bar_color}; width:{compliance_percentage}%; height:100%; border-radius:5px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:5px;">
                <div>0%</div>
                <div>50%</div>
                <div>100%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Combine risk and incident activities
    activities = []

    # Add recent incidents
    if incident_df is not None and not incident_df.empty:
        for _, incident in incident_df.head(3).iterrows():
            timestamp = incident.get("timestamp", "Unknown date")
            description = incident.get("description", "No description")
            severity = incident.get("severity", "unknown").upper()
            activities.append(
                {
                    "date": timestamp,
                    "type": "Incident",
                    "severity": severity,
                    "description": description,
                    "icon": "🚨",  # Alert icon for incidents
                }
            )

    # Add compliance findings as activities
    findings = compliance_results.get("findings", [])
    grouped_findings = format_compliance_findings(findings)

    # Add critical findings first
    for finding in grouped_findings["critical"][:2]:
        activities.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type": "Compliance",
                "severity": "HIGH",
                "description": f"{finding.get('title', 'Issue')}: {finding.get('description', '')}",
                "icon": "⚠️",  # Warning icon for compliance findings
            }
        )

    # Add warning findings
    for finding in grouped_findings["warning"][:1]:
        activities.append(
            {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type": "Compliance",
                "severity": "MEDIUM",
                "description": f"{finding.get('title', 'Warning')}: {finding.get('description', '')}",
                "icon": "⚠️",  # Warning icon for compliance findings
            }
        )

    # Add sample risk mitigations
    sample_mitigations = [
        {
            "date": "2025-05-15 14:30",
            "type": "Mitigation",
            "severity": "HIGH",
            "description": "Implemented demographic fairness monitoring",
            "icon": "🛠️",  # Tools icon for mitigation actions
        },
        {
            "date": "2025-05-10 09:15",
            "type": "Risk Assessment",
            "severity": "MEDIUM",
            "description": "Identified potential age bias in model predictions",
            "icon": "🔍",  # Magnifying glass for assessments
        },
        {
            "date": "2025-05-05 16:45",
            "type": "Documentation",
            "severity": "LOW",
            "description": "Updated Annex IV documentation with new monitoring plan",
            "icon": "📄",  # Document icon for documentation updates
        },
    ]

    # Only add sample mitigations if we don't have enough activities
    if len(activities) < 5:
        activities.extend(sample_mitigations[: 5 - len(activities)])

    # Sort activities by date
    activities = sorted(activities, key=lambda x: x["date"], reverse=True)[:5]

    # Format dates
    activities = format_activities(activities)

    # Display activities as a styled table with icons and better formatting
    style_block = """
    <style>
      .activity-container { margin-top:20px; }
      .activity-row {
        display: flex;
        margin-bottom: 10px;
        padding: 12px 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        transition: box-shadow 0.3s, transform 0.3s;
        border-left: 3px solid #ccc;
      }
      .activity-row:nth-child(odd) { background-color: #ffffff; }
      .activity-row:hover {
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
      }
      .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
      }
      .status-high {
        background-color: #D6404520;
        color: #D64045;
        border: 1px solid #D64045;
      }
      .status-medium {
        background-color: #FFB30F20;
        color: #FFB30F;
        border: 1px solid #FFB30F;
      }
      .status-low {
        background-color: #478C5C20;
        color: #478C5C;
        border: 1px solid #478C5C;
      }
    </style>
    """

    activity_html = style_block + '<div class="activity-container">'

    # Table header (unchanged)
    activity_html += """
      <div style="display:flex; margin-bottom:10px; padding:5px 10px;
                  background-color:#eaeff5; border-radius:5px;
                  font-weight:bold; color:#1f4e79;">
        <div style="flex:0 0 50px;"></div>
        <div style="flex:0 0 180px;">Date</div>
        <div style="flex:0 0 120px;">Type</div>
        <div style="flex:0 0 120px;">Severity</div>
        <div style="flex:1;">Description</div>
      </div>
    """

    # Activity rows with hover effect
    for activity in activities:
        sev = activity["severity"].lower()
        cls = (
            f"status-{sev}"
            if sev in ("high", "medium", "low")
            else "status-medium"
        )
        border = RISK_COLORS.get(activity["severity"], "#999999")

        activity_html += f"""
        <div class="activity-row {cls}" style="border-left:3px solid {border};">
             <div style="flex: 0 0 50px; font-size: 18px; display: flex; align-items: center; justify-content: center;">
                 {activity.get("icon", "📝")}
             </div>
             <div style="flex: 0 0 180px; display: flex; align-items: center;">{activity["formatted_date"]}</div>
             <div style="flex: 0 0 120px; display: flex; align-items: center;">{activity["type"]}</div>
             <div style="flex: 0 0 120px; display: flex; align-items: center;"><span class="status-badge">{activity["severity"]}</span></div>
             <div style="flex: 1; display: flex; align-items: center;">{activity["description"]}</div>
         </div>
        """

    activity_html += "</div>"

    html(activity_html, height=350, scrolling=True)
