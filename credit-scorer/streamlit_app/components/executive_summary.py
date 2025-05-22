# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Executive Summary component for the dashboard."""

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
from src.utils.visualizations.dashboard import (
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


def render_article_card_streamlit(
    article: str, score: float, sources_text: str = "N/A"
) -> str:
    """Render an article card for compliance visualization in Streamlit.

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


def display_exec_summary(risk_df, incident_df):
    """Display an executive summary dashboard."""
    # Add the same CSS styling as the dashboard
    dashboard_css = """
    <style>
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
    </style>
    """

    st.markdown(dashboard_css, unsafe_allow_html=True)

    st.markdown(
        '<div class="card">'
        "<h2>Executive Summary</h2>"
        "<p>Overview of system compliance status, risk indicators, and key metrics related to the EU AI Act requirements.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Get compliance data
    compliance_results = get_compliance_results()
    compliance_summary = get_compliance_summary(compliance_results)

    # Display overall compliance score (matching dashboard)
    if "Overall Compliance" in compliance_summary:
        overall_score = compliance_summary.get("overall_score", 0)
        st.markdown(
            f"""
            <div class="summary-header">
                <h3 style="margin: 0; color:#1F4E79;">Overall Compliance Score</h3>
                <div class="overall-score">{overall_score:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Get detailed compliance information
    article_compliance = compute_article_compliance(
        risk_df, use_compliance_calculator=True
    )
    data_sources = get_compliance_data_sources(compliance_results)
    findings = compliance_results.get("findings", [])
    grouped_findings = format_compliance_findings(findings)

    # Article Cards Grid (NEW LAYOUT - using Streamlit columns)
    st.markdown("<h3>Article Compliance Status</h3>", unsafe_allow_html=True)

    # Create article cards for each article (excluding Overall Compliance)
    article_items = [
        (k, v)
        for k, v in article_compliance.items()
        if k != "Overall Compliance"
    ]

    # Display articles in columns (3 per row)
    cols_per_row = 3
    for i in range(0, len(article_items), cols_per_row):
        cols = st.columns(cols_per_row)

        for j in range(cols_per_row):
            if i + j < len(article_items):
                article, score = article_items[i + j]

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

                # Determine color based on score
                if score >= 80:
                    color = "#478C5C"  # green
                    bg_color = "#C6E0B4"
                elif score >= 60:
                    color = "#FFB30F"  # yellow
                    bg_color = "#FFE699"
                else:
                    color = "#D64045"  # red
                    bg_color = "#F8CBAD"

                with cols[j]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: white;
                            border-radius: 8px;
                            padding: 15px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                            margin-bottom: 15px;
                            min-height: 120px;
                        ">
                            <div style="
                                font-weight: bold;
                                margin-bottom: 5px;
                                font-size: 14px;
                                color: #1F4E79;
                            ">{article}</div>
                            <div style="
                                font-size: 24px;
                                font-weight: bold;
                                margin-bottom: 10px;
                                color: {color};
                            ">{score:.1f}%</div>
                            <div style="
                                height: 8px;
                                background-color: #e0e0e0;
                                border-radius: 4px;
                                overflow: hidden;
                                margin-bottom: 5px;
                            ">
                                <div style="
                                    height: 100%;
                                    border-radius: 4px;
                                    width: {score}%;
                                    background-color: {bg_color};
                                "></div>
                            </div>
                            <div style="
                                font-size: 11px;
                                color: #666;
                            "><strong>Sources:</strong> {sources_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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

    # Risk Metrics Section (MOVED BELOW and matching dashboard exactly)
    st.markdown(
        """
        <div class="metrics-section">
            <div class="metrics-header">
                <h3 style="margin: 0; color:#1F4E79;">Risk & Compliance Metrics</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Risk Metrics Section (using Streamlit columns instead of HTML grid)
    st.markdown("<h3>Risk & Compliance Metrics</h3>", unsafe_allow_html=True)

    # Display metrics in 4 columns per row (2 rows total)
    # First row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{total_risks}</div>
                <div style="font-size: 14px; color: #666;">Risks Identified</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{critical_findings}</div>
                <div style="font-size: 14px; color: #666;">Critical Findings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{warning_findings}</div>
                <div style="font-size: 14px; color: #666;">Warning Findings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{completion_status:.0%}</div>
                <div style="font-size: 14px; color: #666;">Mitigation Progress</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Second row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{high_risks}</div>
                <div style="font-size: 14px; color: #666;">High Risk Items</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px;">{total_incidents}</div>
                <div style="font-size: 14px; color: #666;">Reported Incidents</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                border-left: 4px solid #478C5C;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px; color: #478C5C;">{strongest_article["score"]:.0f}%</div>
                <div style="font-size: 14px; color: #666;">Strongest: {strongest_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style="
                background-color: white;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                text-align: center;
                border-left: 4px solid #D64045;
                min-height: 80px;
            ">
                <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px; color: #D64045;">{weakest_article["score"]:.0f}%</div>
                <div style="font-size: 14px; color: #666;">Weakest: {weakest_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Findings Section (using Streamlit native components)
    st.markdown("<h3>Key Compliance Findings</h3>", unsafe_allow_html=True)

    findings_displayed = False

    # Helper function to deduplicate findings
    def deduplicate_findings(findings_list):
        seen = set()
        unique_findings = []
        for finding in findings_list:
            # Create a unique key based on title and description
            title = finding.get("title", finding.get("message", "Finding"))
            description = finding.get(
                "description", finding.get("recommendation", "")
            )
            key = f"{title}|{description}"
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)
        return unique_findings

    # Critical issues first
    if grouped_findings["critical"]:
        unique_critical = deduplicate_findings(grouped_findings["critical"])
        st.markdown("**🚨 Critical Issues:**", unsafe_allow_html=True)
        findings_displayed = True
        for finding in unique_critical[:3]:  # Show top 3 unique findings
            title = finding.get("title", finding.get("message", "Finding"))
            description = finding.get(
                "description", finding.get("recommendation", "")
            )
            st.markdown(
                f"""
                <div style="
                    padding: 12px;
                    margin-bottom: 5px;
                    background-color: #ffecec;
                    border-left: 3px solid #D64045;
                    border-radius: 5px;
                ">
                    <div style="font-weight: bold; margin-bottom: 5px;">{title}</div>
                    <div>{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Warnings next
    if grouped_findings["warning"]:
        unique_warnings = deduplicate_findings(grouped_findings["warning"])
        st.markdown("**⚠️ Warnings:**", unsafe_allow_html=True)
        findings_displayed = True
        for finding in unique_warnings[:3]:  # Show top 3 unique findings
            title = finding.get("title", finding.get("message", "Finding"))
            description = finding.get(
                "description", finding.get("recommendation", "")
            )
            st.markdown(
                f"""
                <div style="
                    padding: 12px;
                    margin-bottom: 5px;
                    background-color: #fff9ec;
                    border-left: 3px solid #FFB30F;
                    border-radius: 5px;
                ">
                    <div style="font-weight: bold; margin-bottom: 5px;">{title}</div>
                    <div>{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Positive findings
    if grouped_findings["positive"]:
        unique_positive = deduplicate_findings(grouped_findings["positive"])
        st.markdown("**✅ Positive Findings:**", unsafe_allow_html=True)
        findings_displayed = True
        for finding in unique_positive[:3]:  # Show top 3 unique findings
            title = finding.get("title", finding.get("message", "Finding"))
            description = finding.get(
                "description", finding.get("recommendation", "")
            )
            st.markdown(
                f"""
                <div style="
                    padding: 12px;
                    margin-bottom: 5px;
                    background-color: #f0f7f0;
                    border-left: 3px solid #478C5C;
                    border-radius: 5px;
                ">
                    <div style="font-weight: bold; margin-bottom: 5px;">{title}</div>
                    <div>{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # If no findings, show a message
    if not findings_displayed:
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
