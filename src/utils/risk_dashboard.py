"""Streamlit dashboard for visualizing risk register information.

Provides non-technical stakeholders with visibility into model risks.

Usage:
    streamlit run src/utils/risk_dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RISK_REGISTER_PATH = Path("compliance/risk/risk_register.xlsx")


def load_risk_register():
    """Load the risk register from the default location or most recent file."""
    risk_register_path = Path(RISK_REGISTER_PATH)

    try:
        excel_data = pd.read_excel(risk_register_path, sheet_name=None)

        # Process each sheet - normalize column names to lowercase
        for sheet_name, df in excel_data.items():
            excel_data[sheet_name].columns = [col.lower() for col in df.columns]

        return excel_data
    except Exception as e:
        st.error(f"Error loading risk register: {e}")
        return None


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Credit Scoring Model Risk Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Credit Scoring Model - Risk Dashboard")
    st.markdown("### EU AI Act Compliance: Article 9 (Risk Management)")

    # Load risk register - now contains all sheets
    excel_data = load_risk_register()

    if excel_data is None:
        st.stop()

    # Create tabs to switch between sheets
    available_sheets = list(excel_data.keys())

    if len(available_sheets) > 1:
        selected_tab = st.tabs(available_sheets)

        # First tab - main Risks sheet
        with selected_tab[0]:
            display_risks_dashboard(excel_data["Risks"])

        # Second tab - HazardDetails if available
        if len(available_sheets) > 1 and "HazardDetails" in available_sheets:
            with selected_tab[1]:
                display_hazard_details(excel_data["HazardDetails"])
    else:
        # Fallback if only one sheet
        display_risks_dashboard(excel_data[available_sheets[0]])


def display_risks_dashboard(risk_df):
    """Display the main risks dashboard."""
    # Dashboard layout with columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk Level Distribution")
        # Use risk_category (previously called risk_level in the dashboard but may be called 'risk_category' in db)
        severity_column = next(
            (col for col in ["risk_category", "risk_level"] if col in risk_df.columns), None
        )

        if severity_column:
            risk_counts = risk_df[severity_column].value_counts()
            colors = {
                "HIGH": "#FF5733",
                "MEDIUM": "#FFC300",
                "LOW": "#33FF57",
            }
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map=colors,
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No risk levels found in the risk register.")

    with col2:
        st.subheader("Risk Category By Run")
        # Let's use run_id for the second chart to show risk distribution across different runs
        if "run_id" in risk_df.columns and severity_column:
            run_risk_counts = (
                risk_df.groupby(["run_id", severity_column]).size().reset_index(name="count")
            )
            fig = px.bar(
                run_risk_counts,
                x="run_id",
                y="count",
                color=severity_column,
                color_discrete_map={"HIGH": "#FF5733", "MEDIUM": "#FFC300", "LOW": "#33FF57"},
                title="",
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Missing required columns for run-level analysis.")

    # Risk metrics overview
    st.subheader("Risk Metrics")
    metrics_cols = st.columns(3)

    with metrics_cols[0]:
        if "risk_overall" in risk_df.columns:
            avg_risk = risk_df["risk_overall"].mean()
            st.metric("Average Risk Score", f"{avg_risk:.2f}")
        else:
            st.info("No risk scores found.")

    with metrics_cols[1]:
        if severity_column:
            high_risks = (risk_df[severity_column] == "HIGH").sum()
            st.metric("High Risk Items", high_risks)
        else:
            st.info("No risk levels found.")

    with metrics_cols[2]:
        if "status" in risk_df.columns:
            mitigated = (risk_df["status"] == "COMPLETED").sum()
            total = len(risk_df)
            st.metric("Mitigation Progress", f"{mitigated}/{total}")
        else:
            st.info("No status information found.")

    # Filters and interactive elements
    st.subheader("Risk Register Details")

    # Filters
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
        if "run_id" in risk_df.columns:
            run_id_filter = st.multiselect(
                "Filter by Run ID",
                options=sorted(risk_df["run_id"].unique().tolist()),
                default=sorted(risk_df["run_id"].unique().tolist()),
            )
        else:
            run_id_filter = []

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

    if run_id_filter and "run_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["run_id"].isin(run_id_filter)]

    if status_filter and "status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]

    # Detailed risk table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400,
    )

    # Mitigation progress tracking
    st.subheader("Mitigation Progress")

    if all(col in risk_df.columns for col in ["risk_description", "mitigation", "status"]):
        # Calculate mitigation progress
        completed = (risk_df["status"] == "COMPLETED").sum()
        total = len(risk_df)
        progress = completed / total if total > 0 else 0

        st.progress(progress)
        st.write(f"**{completed}** out of **{total}** mitigations completed ({progress:.1%})")

        # Display mitigation status by severity
        if severity_column and "status" in risk_df.columns:
            mitigation_status = pd.crosstab(risk_df[severity_column], risk_df["status"])

            fig = px.bar(
                mitigation_status, barmode="group", title="Mitigation Status by Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Mitigation information not found in the risk register.")

    # Export options
    st.subheader("Export Options")
    export_cols = st.columns([1, 1])

    with export_cols[0]:
        export_format = st.selectbox(
            "Export Format",
            options=["Excel (.xlsx)", "CSV (.csv)"],
            key="risk_register_export_format",
        )

    with export_cols[1]:
        import io

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


def display_hazard_details(hazard_df):
    """Display the hazard details dashboard."""
    st.subheader("Hazard Details")
    st.write(
        "This view shows detailed information about specific hazards identified during risk assessment."
    )

    # Filters for hazard details
    filter_cols = st.columns(3)

    with filter_cols[0]:
        if "severity" in hazard_df.columns:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=sorted(hazard_df["severity"].unique().tolist()),
                default=sorted(hazard_df["severity"].unique().tolist()),
            )
        else:
            severity_filter = []

    with filter_cols[1]:
        if "hazard_id" in hazard_df.columns:
            hazard_id_filter = st.multiselect(
                "Filter by Hazard Type",
                options=sorted(hazard_df["hazard_id"].unique().tolist()),
                default=sorted(hazard_df["hazard_id"].unique().tolist()),
            )
        else:
            hazard_id_filter = []

    with filter_cols[2]:
        if "run_id" in hazard_df.columns:
            run_id_filter = st.multiselect(
                "Filter by Run ID",
                options=sorted(hazard_df["run_id"].unique().tolist()),
                default=sorted(hazard_df["run_id"].unique().tolist()),
            )
        else:
            run_id_filter = []

    # Apply filters
    filtered_df = hazard_df.copy()

    if severity_filter and "severity" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["severity"].isin(severity_filter)]

    if hazard_id_filter and "hazard_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["hazard_id"].isin(hazard_id_filter)]

    if run_id_filter and "run_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["run_id"].isin(run_id_filter)]

    # Display hazard details table
    st.dataframe(filtered_df, use_container_width=True, height=400)

    # Hazard type distribution
    if "hazard_id" in hazard_df.columns:
        st.subheader("Hazard Type Distribution")
        hazard_counts = hazard_df["hazard_id"].value_counts()
        fig = px.pie(
            values=hazard_counts.values,
            names=hazard_counts.index,
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Export options for hazard details
    st.subheader("Export Options")
    export_cols = st.columns([1, 1])

    with export_cols[0]:
        export_format = st.selectbox(
            "Export Format",
            options=["Excel (.xlsx)", "CSV (.csv)"],
            key="hazard_export_format",  # Unique key to avoid conflict with other selectbox
        )

    with export_cols[1]:
        st.write("")  # Vertical spacing for alignment
        if st.button(
            "Export Filtered Data",
            help="Export the hazard data with current filters applied",
            key="hazard_export_button",  # Unique key to avoid conflict
        ):
            export_dir = Path("compliance/exports")
            export_dir.mkdir(exist_ok=True, parents=True)

            if export_format == "Excel (.xlsx)":
                export_path = export_dir / "hazard_details_export.xlsx"
                filtered_df.to_excel(export_path, index=False)
            else:  # CSV
                export_path = export_dir / "hazard_details_export.csv"
                filtered_df.to_csv(export_path, index=False)

            st.success(f"Data exported to {export_path}")


if __name__ == "__main__":
    main()
