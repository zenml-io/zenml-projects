"""Streamlit dashboard for visualizing risk register information and Annex IV documentation.

Provides non-technical stakeholders with visibility into model risks and compliance documentation.

Usage:
    streamlit run dashboard.py
"""

import json
import tempfile
from pathlib import Path

import markdown
import pandas as pd
import pdfkit
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html

# Constants
RISK_REGISTER_PATH = Path("docs/risk/risk_register.xlsx")
RELEASES_DIR = Path("docs/releases")
TEMPLATE_DIR = Path("docs/templates")
SAMPLE_INPUTS_PATH = Path("docs/templates/sample_inputs.json")


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


def load_latest_annex_iv():
    """Load the most recent Annex IV document."""
    try:
        # Find the most recent release directory
        release_dirs = sorted(
            [d for d in RELEASES_DIR.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not release_dirs:
            st.warning("No release directories found.")
            return None, None

        latest_release = release_dirs[0]

        # Find the Annex IV document
        annex_files = list(latest_release.glob("annex_iv_*.md"))
        if not annex_files:
            st.warning(f"No Annex IV document found in {latest_release}.")
            return None, None

        # Load the first Annex IV document found
        annex_path = annex_files[0]
        with open(annex_path, "r") as f:
            content = f.read()

        return content, annex_path

    except Exception as e:
        st.error(f"Error loading Annex IV document: {e}")
        return None, None


def save_manual_inputs(manual_inputs):
    """Save the manual inputs to the sample inputs file."""
    try:
        with open(SAMPLE_INPUTS_PATH, "w") as f:
            json.dump(manual_inputs, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving manual inputs: {e}")
        return False


def export_annex_iv_to_pdf(markdown_content, output_path=None):
    """Export the Annex IV document to PDF with multiple fallback options."""
    try:
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # Add some basic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Annex IV Documentation</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 2cm;
                    font-size: 12px;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #3498db;
                    margin-top: 20px;
                }}
                h3 {{
                    color: #2980b9;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                img {{
                    max-width: 100%;
                }}
                code {{
                    background-color: #f8f8f8;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    font-family: monospace;
                    padding: 2px 4px;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Generate PDF using pdfkit (which requires wkhtmltopdf)
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

        # Try pdfkit first
        try:
            options = {
                "page-size": "A4",
                "margin-top": "1cm",
                "margin-right": "1cm",
                "margin-bottom": "1cm",
                "margin-left": "1cm",
                "encoding": "UTF-8",
            }
            pdfkit.from_string(styled_html, output_path, options=options)
            return output_path
        except Exception as pdfkit_error:
            st.warning(f"pdfkit failed: {pdfkit_error}")

            # Fallback to weasyprint
            try:
                import weasyprint

                html_doc = weasyprint.HTML(string=styled_html)
                html_doc.write_pdf(output_path)
                st.info("PDF generated using weasyprint fallback.")
                return output_path
            except ImportError:
                st.error("weasyprint not installed. Install with: pip install weasyprint")
                raise
            except Exception as weasyprint_error:
                st.warning(f"weasyprint also failed: {weasyprint_error}")

                # Final fallback - save as HTML
                html_output_path = output_path.replace(".pdf", ".html")
                with open(html_output_path, "w", encoding="utf-8") as f:
                    f.write(styled_html)
                st.warning(f"Could not generate PDF. Saved as HTML instead: {html_output_path}")
                return html_output_path

    except Exception as e:
        st.error(f"Error exporting to PDF: {e}")
        st.info("Note: PDF export requires wkhtmltopdf to be installed.")
        return None


def parse_requirements_txt(requirements_path="requirements.txt"):
    """Parse requirements.txt and extract package versions."""
    frameworks = {}

    try:
        with open(requirements_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse package==version or package>=version patterns
            if "==" in line:
                package, version = line.split("==", 1)
                frameworks[package.strip()] = version.strip()
            elif ">=" in line:
                package, version = line.split(">=", 1)
                frameworks[package.strip()] = f">={version.strip()}"
            elif "~=" in line:
                package, version = line.split("~=", 1)
                frameworks[package.strip()] = f"~={version.strip()}"
            else:
                # Handle cases like "package" without version
                package = line
                frameworks[package.strip()] = "latest"

    except FileNotFoundError:
        st.warning(f"requirements.txt not found at {requirements_path}")
        return {}
    except Exception as e:
        st.error(f"Error parsing requirements.txt: {e}")
        return {}

    return frameworks


def load_manual_inputs():
    """Load the manual inputs for the Annex IV document."""
    try:
        with open(SAMPLE_INPUTS_PATH, "r") as f:
            manual_inputs = json.load(f)

        # Auto-populate frameworks from requirements.txt if not already set
        if "frameworks" not in manual_inputs or not manual_inputs["frameworks"]:
            manual_inputs["frameworks"] = parse_requirements_txt()

        return manual_inputs
    except Exception as e:
        st.error(f"Error loading manual inputs: {e}")
        return {}


def render_markdown_with_newlines(content):
    """Convert JSON content with \n to proper markdown with line breaks."""
    # Replace \n with actual newlines for proper markdown rendering
    if isinstance(content, str):
        return content.replace("\\n", "\n")
    elif isinstance(content, dict):
        return {k: render_markdown_with_newlines(v) for k, v in content.items()}
    elif isinstance(content, list):
        return [render_markdown_with_newlines(item) for item in content]
    else:
        return content


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
            export_dir = Path("docs/exports")
            export_dir.mkdir(exist_ok=True, parents=True)

            if export_format == "Excel (.xlsx)":
                export_path = export_dir / "hazard_details_export.xlsx"
                filtered_df.to_excel(export_path, index=False)
            else:  # CSV
                export_path = export_dir / "hazard_details_export.csv"
                filtered_df.to_csv(export_path, index=False)

            st.success(f"Data exported to {export_path}")


def display_annex_iv_documentation():
    """Display and allow editing of the Annex IV documentation."""
    st.subheader("Annex IV Documentation")

    # Load the latest Annex IV document and manual inputs
    annex_content, annex_path = load_latest_annex_iv()
    manual_inputs = load_manual_inputs()

    if annex_content is None:
        st.warning("No Annex IV documentation found.")
        return

    # Process the content to handle newlines properly
    processed_content = render_markdown_with_newlines(annex_content)

    # Tabs for viewing and editing
    view_tab, edit_tab, frameworks_tab = st.tabs(
        ["View Documentation", "Edit Fields", "Manage Frameworks"]
    )

    with view_tab:
        # Display the Annex IV document with proper newline handling
        st.markdown(processed_content)

        # Export options
        st.subheader("Export Options")

        # PDF export
        if st.button("Export to PDF", key="annex_iv_pdf_export"):
            pdf_path = export_annex_iv_to_pdf(processed_content)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="annex_iv_documentation.pdf",
                    mime="application/pdf",
                    key="download_annex_iv_pdf",
                )

    with edit_tab:
        st.info(
            "Edit the fields below to update the Annex IV documentation. "
            "Changes will be used the next time the Annex IV documentation is generated."
        )

        # Group inputs into logical sections
        st.subheader("General Information")
        manual_inputs["provider"] = st.text_input("Provider", manual_inputs.get("provider", ""))
        manual_inputs["description"] = st.text_area(
            "Description", manual_inputs.get("description", "")
        )
        manual_inputs["intended_purpose"] = st.text_area(
            "Intended Purpose", manual_inputs.get("intended_purpose", "")
        )

        st.subheader("System Architecture")
        manual_inputs["product_image_url"] = st.text_input(
            "Product Image URL", manual_inputs.get("product_image_url", "")
        )
        manual_inputs["ui_screenshot_url"] = st.text_input(
            "UI Screenshot URL", manual_inputs.get("ui_screenshot_url", "")
        )
        manual_inputs["arch_diagram_url"] = st.text_input(
            "Architecture Diagram URL", manual_inputs.get("arch_diagram_url", "")
        )
        manual_inputs["hardware_requirements"] = st.text_area(
            "Hardware Requirements", manual_inputs.get("hardware_requirements", "")
        )

        st.subheader("Model Information")
        manual_inputs["model_architecture"] = st.text_input(
            "Model Architecture", manual_inputs.get("model_architecture", "")
        )
        manual_inputs["optimization_objective"] = st.text_area(
            "Optimization Objective", manual_inputs.get("optimization_objective", "")
        )

        # Performance metrics as a JSON editor
        st.subheader("Performance Metrics")
        perf_metrics_json = json.dumps(manual_inputs.get("performance_metrics", {}), indent=2)
        perf_metrics_edited = st.text_area(
            "Performance Metrics (JSON)", perf_metrics_json, height=200
        )
        try:
            manual_inputs["performance_metrics"] = json.loads(perf_metrics_edited)
        except json.JSONDecodeError:
            st.error("Invalid JSON format for performance metrics")

        # Fairness assessment as a JSON editor
        st.subheader("Fairness Assessment")
        fairness_json = json.dumps(manual_inputs.get("fairness_assessment", {}), indent=2)
        fairness_edited = st.text_area("Fairness Assessment (JSON)", fairness_json, height=200)
        try:
            manual_inputs["fairness_assessment"] = json.loads(fairness_edited)
        except json.JSONDecodeError:
            st.error("Invalid JSON format for fairness assessment")

        st.subheader("Risk and Compliance")
        # Use text_area for multi-line fields to preserve newlines
        manual_inputs["risk_management_system"] = st.text_area(
            "Risk Management System",
            manual_inputs.get("risk_management_system", ""),
            help="Use line breaks to separate different points",
        )
        manual_inputs["lifecycle_changes_log"] = st.text_area(
            "Lifecycle Changes Log",
            manual_inputs.get("lifecycle_changes_log", ""),
            help="Use line breaks to separate different versions",
        )
        manual_inputs["declaration_of_conformity"] = st.text_area(
            "EU Declaration of Conformity",
            manual_inputs.get("declaration_of_conformity", ""),
            height=300,
            help="Use line breaks to format the declaration properly",
        )

        # Save changes
        if st.button("Save Changes", key="save_annex_iv_changes"):
            if save_manual_inputs(manual_inputs):
                st.success(
                    "Changes saved successfully. They will be applied the next time the Annex IV documentation is generated."
                )
            else:
                st.error("Failed to save changes.")

    with frameworks_tab:
        st.subheader("Framework Versions")
        st.info(
            "Framework versions are automatically loaded from requirements.txt. You can override them here if needed."
        )

        # Show current frameworks from requirements.txt
        current_frameworks = parse_requirements_txt()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**From requirements.txt:**")
            if current_frameworks:
                for package, version in current_frameworks.items():
                    st.code(f"{package}: {version}")
            else:
                st.warning("No frameworks found in requirements.txt")

        with col2:
            st.write("**Current overrides:**")
            user_frameworks = manual_inputs.get("frameworks", {})

            # Allow users to add/edit framework versions
            st.write("Add or override framework versions:")

            # Create input fields for existing user frameworks
            for package, version in user_frameworks.items():
                new_version = st.text_input(f"{package}", value=version, key=f"fw_{package}")
                user_frameworks[package] = new_version

            # Add new framework input
            col_new1, col_new2 = st.columns(2)
            with col_new1:
                new_package = st.text_input("New Package Name", key="new_package")
            with col_new2:
                new_version = st.text_input("Version", key="new_version")

            if st.button("Add Framework") and new_package and new_version:
                user_frameworks[new_package] = new_version
                st.success(f"Added {new_package}: {new_version}")

            # Update the frameworks in manual_inputs
            manual_inputs["frameworks"] = user_frameworks

            # Button to refresh from requirements.txt
            if st.button("Refresh from requirements.txt"):
                manual_inputs["frameworks"] = parse_requirements_txt()
                st.success("Refreshed frameworks from requirements.txt")
                st.experimental_rerun()


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Credit Scoring Model - EU AI Act Compliance Dashboard",
        page_icon="📊",
        layout="wide",
    )

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Risk Dashboard", "Annex IV Documentation"],
    )

    if page == "Risk Dashboard":
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

    elif page == "Annex IV Documentation":
        st.title("📝 Annex IV Technical Documentation")
        st.markdown("### EU AI Act Compliance: Article 11 (Technical Documentation)")

        display_annex_iv_documentation()


if __name__ == "__main__":
    main()
