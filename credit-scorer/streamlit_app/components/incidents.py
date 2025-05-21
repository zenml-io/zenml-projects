"""Incident Tracking component for the dashboard."""

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_app.config import PRIMARY_COLOR, RISK_COLORS, SECONDARY_COLOR


def display_incident_tracking(incident_df):
    """Display the incident tracking dashboard."""
    st.markdown(
        '<div class="card">'
        "<h2>Incident Management</h2>"
        "<p>Tracking and resolution of incidents as required by EU AI Act Articles 18-19.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if incident_df is None or incident_df.empty:
        st.info("No incidents recorded in the system.")
        return

    # Incident metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        total_incidents = len(incident_df)
        st.markdown(
            '<div class="metric-card">'
            f'<div class="metric-value">{total_incidents}</div>'
            '<div class="metric-label">Total Incidents</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col2:
        if "severity" in incident_df.columns:
            high_severity = (incident_df["severity"] == "high").sum()
            st.markdown(
                '<div class="metric-card">'
                f'<div class="metric-value">{high_severity}</div>'
                '<div class="metric-label">High Severity</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value">N/A</div>'
                '<div class="metric-label">High Severity</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    with col3:
        if "resolved" in incident_df.columns:
            resolved = incident_df["resolved"].sum() if "resolved" in incident_df.columns else 0
            resolution_rate = (resolved / total_incidents) if total_incidents > 0 else 0
            st.markdown(
                '<div class="metric-card">'
                f'<div class="metric-value">{resolution_rate:.0%}</div>'
                '<div class="metric-label">Resolution Rate</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="metric-card">'
                '<div class="metric-value">N/A</div>'
                '<div class="metric-label">Resolution Rate</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    # Incident visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Incidents by Severity</h3>", unsafe_allow_html=True)

        if "severity" in incident_df.columns:
            severity_counts = incident_df["severity"].value_counts()

            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                color=severity_counts.index,
                color_discrete_map={
                    "high": RISK_COLORS["HIGH"],
                    "medium": RISK_COLORS["MEDIUM"],
                    "low": RISK_COLORS["LOW"],
                },
                hole=0.6,
            )
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=0,
                    xanchor="center",
                    x=0.5,
                ),
                height=300,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No severity information available in incident data.")

    with col2:
        st.markdown("<h3>Incidents by Source</h3>", unsafe_allow_html=True)

        if "source" in incident_df.columns:
            source_counts = incident_df["source"].value_counts()

            fig = px.bar(
                x=source_counts.index,
                y=source_counts.values,
                color=source_counts.index,
                color_discrete_sequence=[
                    PRIMARY_COLOR,
                    SECONDARY_COLOR,
                    "#999999",
                ],
            )
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=False,
                xaxis_title="Source",
                yaxis_title="Count",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No source information available in incident data.")

    # Incident list
    st.markdown("<h3>Recorded Incidents</h3>", unsafe_allow_html=True)

    # Filters
    with st.expander("Filters", expanded=False):
        filter_cols = st.columns(3)

        with filter_cols[0]:
            if "severity" in incident_df.columns:
                severity_options = incident_df["severity"].unique().tolist()
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=severity_options,
                    default=severity_options,
                )
            else:
                severity_filter = []

        with filter_cols[1]:
            if "source" in incident_df.columns:
                source_options = incident_df["source"].unique().tolist()
                source_filter = st.multiselect(
                    "Filter by Source",
                    options=source_options,
                    default=source_options,
                )
            else:
                source_filter = []

        with filter_cols[2]:
            if "timestamp" in incident_df.columns:
                # Convert timestamp strings to datetime objects
                incident_df["timestamp"] = pd.to_datetime(incident_df["timestamp"], errors="coerce")

                min_date = (
                    incident_df["timestamp"].min().date()
                    if not incident_df["timestamp"].isna().all()
                    else datetime.now().date()
                )
                max_date = (
                    incident_df["timestamp"].max().date()
                    if not incident_df["timestamp"].isna().all()
                    else datetime.now().date()
                )

                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
            else:
                date_range = None

    # Apply filters
    filtered_df = incident_df.copy()

    if severity_filter and "severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["severity"].isin(severity_filter)]

    if source_filter and "source" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["source"].isin(source_filter)]

    if date_range and len(date_range) == 2 and "timestamp" in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["timestamp"].dt.date >= start_date)
            & (filtered_df["timestamp"].dt.date <= end_date)
        ]

    # Display incident table with formatting
    if not filtered_df.empty:
        # Create a custom display DataFrame with selected columns
        display_columns = []
        if "timestamp" in filtered_df.columns:
            display_columns.append("timestamp")
        if "severity" in filtered_df.columns:
            display_columns.append("severity")
        if "description" in filtered_df.columns:
            display_columns.append("description")
        if "source" in filtered_df.columns:
            display_columns.append("source")
        if "resolved" in filtered_df.columns:
            display_columns.append("resolved")

        # Ensure we have some columns
        if not display_columns:
            display_columns = filtered_df.columns.tolist()

        display_df = filtered_df[display_columns].copy()

        # Format timestamp if present
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "severity": st.column_config.Column(
                    "Severity",
                    help="Incident severity level",
                    width="medium",
                ),
                "description": st.column_config.Column(
                    "Description",
                    help="Incident description",
                    width="large",
                ),
                "resolved": st.column_config.CheckboxColumn(
                    "Resolved",
                    help="Whether the incident has been resolved",
                ),
            },
        )
    else:
        st.info("No incidents match the selected filters.")
