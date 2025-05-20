"""Risk Management component for the dashboard."""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.config import PRIMARY_COLOR, RISK_COLORS, SECONDARY_COLOR


def display_risks_dashboard(risk_df):
    """Display the main risks dashboard."""
    st.markdown(
        '<div class="card">'
        "<h2>Risk Management</h2>"
        "<p>Monitoring and management of risks as required by EU AI Act Article 9.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Dashboard layout with columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("<h3>Risk Level Distribution</h3>", unsafe_allow_html=True)
        # Use risk_category (previously called risk_level in the dashboard but may be called 'risk_category' in db)
        severity_column = next(
            (col for col in ["risk_category", "risk_level"] if col in risk_df.columns), None
        )

        if severity_column:
            risk_counts = risk_df[severity_column].value_counts()
            colors = {
                "HIGH": RISK_COLORS["HIGH"],
                "MEDIUM": RISK_COLORS["MEDIUM"],
                "LOW": RISK_COLORS["LOW"],
            }
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map=colors,
                hole=0.6,
            )
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5),
                height=300,
                font=dict(family="Arial", size=12),
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk levels found in the risk register.")

    with col2:
        st.markdown("<h3>Risk Distribution by Category</h3>", unsafe_allow_html=True)

        if "category" in risk_df.columns and severity_column:
            # Create a risk heatmap by category and risk level
            category_severity = pd.crosstab(risk_df["category"], risk_df[severity_column])

            # Set correct order for risk levels
            if all(level in category_severity.columns for level in ["HIGH", "MEDIUM", "LOW"]):
                category_severity = category_severity[["HIGH", "MEDIUM", "LOW"]]

            # Create heatmap
            fig = px.imshow(
                category_severity,
                color_continuous_scale=["#e6f7ff", "#4169E1"],
                aspect="auto",
                text_auto=True,
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                coloraxis_showscale=False,
                height=300,
                xaxis_title="Risk Level",
                yaxis_title="Risk Category",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to risk level by run if category not available
            if "run_id" in risk_df.columns and severity_column:
                run_risk_counts = (
                    risk_df.groupby(["run_id", severity_column]).size().reset_index(name="count")
                )
                fig = px.bar(
                    run_risk_counts,
                    x="run_id",
                    y="count",
                    color=severity_column,
                    color_discrete_map=RISK_COLORS,
                    title="",
                )
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                    height=300,
                    xaxis_title="Pipeline Run",
                    yaxis_title="Number of Risks",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing required columns for detailed risk analysis.")

    # Filters and interactive elements
    st.markdown("<h3>Risk Register</h3>", unsafe_allow_html=True)

    # Filters
    with st.expander("Filters", expanded=False):
        filter_cols = st.columns(3)

        with filter_cols[0]:
            if severity_column:
                risk_level_filter = st.multiselect(
                    "Filter by Risk Level",
                    options=sorted(risk_df[severity_column].unique().tolist()),
                    default=sorted(risk_df[severity_column].unique().tolist()),
                )
            else:
                risk_level_filter = []

        with filter_cols[1]:
            if "category" in risk_df.columns:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=sorted(risk_df["category"].unique().tolist()),
                    default=sorted(risk_df["category"].unique().tolist()),
                )
            elif "run_id" in risk_df.columns:
                category_filter = st.multiselect(
                    "Filter by Run ID",
                    options=sorted(risk_df["run_id"].unique().tolist()),
                    default=sorted(risk_df["run_id"].unique().tolist()),
                )
            else:
                category_filter = []

        with filter_cols[2]:
            if "status" in risk_df.columns:
                status_filter = st.multiselect(
                    "Filter by Mitigation Status",
                    options=sorted(risk_df["status"].unique().tolist()),
                    default=sorted(risk_df["status"].unique().tolist()),
                )
            else:
                status_filter = []

    # Apply filters
    filtered_df = risk_df.copy()

    if risk_level_filter and severity_column:
        filtered_df = filtered_df[filtered_df[severity_column].isin(risk_level_filter)]

    if category_filter:
        if "category" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]
        elif "run_id" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["run_id"].isin(category_filter)]

    if status_filter and "status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]

    # Detailed risk table with styled status column
    if not filtered_df.empty:
        # Select relevant columns and reorder for better presentation
        columns_to_show = []
        if "id" in filtered_df.columns:
            columns_to_show.append("id")
        if "risk_description" in filtered_df.columns:
            columns_to_show.append("risk_description")
        if severity_column:
            columns_to_show.append(severity_column)
        if "category" in filtered_df.columns:
            columns_to_show.append("category")
        if "status" in filtered_df.columns:
            columns_to_show.append("status")
        if "mitigation" in filtered_df.columns:
            columns_to_show.append("mitigation")

        # Ensure we have some columns
        if not columns_to_show:
            columns_to_show = filtered_df.columns.tolist()

        display_df = filtered_df[columns_to_show].copy()

        # Add styling to the dataframe
        # Note: This is limited in Streamlit, but we can customize to some extent
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                severity_column: st.column_config.Column(
                    "Risk Level",
                    help="Severity of the risk",
                    width="medium",
                ),
                "risk_description": st.column_config.Column(
                    "Description",
                    help="Risk description",
                    width="large",
                ),
                "status": st.column_config.Column(
                    "Status",
                    help="Mitigation status",
                    width="medium",
                ),
            }
            if severity_column
            else None,
        )
    else:
        st.info("No risks match the selected filters.")

    # Mitigation progress tracking
    st.markdown("<h3>Mitigation Progress</h3>", unsafe_allow_html=True)

    if all(col in risk_df.columns for col in ["risk_description", "mitigation", "status"]):
        # Calculate mitigation progress
        completed = (risk_df["status"] == "COMPLETED").sum()
        total = len(risk_df)
        progress = completed / total if total > 0 else 0

        col1, col2 = st.columns([3, 1])

        with col1:
            # More visually appealing progress bar
            st.progress(progress, f"Progress: {progress:.1%}")

        with col2:
            st.markdown(
                f'<div style="text-align: center;"><span style="font-size: 2rem; font-weight: bold; color: {PRIMARY_COLOR};">{completed}</span> / {total}</div>',
                unsafe_allow_html=True,
            )

        # Display mitigation status by severity
        if severity_column and "status" in risk_df.columns:
            mitigation_status = pd.crosstab(risk_df[severity_column], risk_df["status"])

            fig = px.bar(
                mitigation_status,
                barmode="group",
                color_discrete_sequence=[PRIMARY_COLOR, SECONDARY_COLOR, "#999999"],
                height=300,
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                xaxis_title="Risk Level",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Mitigation information not found in the risk register.")

    # Export options
    with st.expander("Export Options", expanded=False):
        export_cols = st.columns([1, 1])

        with export_cols[0]:
            export_format = st.selectbox(
                "Export Format",
                options=["Excel (.xlsx)", "CSV (.csv)"],
                key="risk_register_export_format",
            )

        with export_cols[1]:
            st.write("")  # empty space to align the button with the dropdown
            if export_format == "Excel (.xlsx)":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name="Sheet1")
                buffer.seek(0)
                download_data = buffer
                file_name = "risk_register_export.xlsx"
                mime_type = "application/vnd.ms-excel"
            else:  # CSV
                buffer = io.BytesIO()
                filtered_df.to_csv(buffer, index=False)
                buffer.seek(0)
                download_data = buffer
                file_name = "risk_register_export.csv"
                mime_type = "text/csv"

            # Create download button
            st.download_button(
                label="Download Filtered Data",
                data=download_data,
                file_name=file_name,
                mime=mime_type,
                help="Download the data with current filters applied",
                key="risk_register_download_button",
            )
