"""Annex IV Documentation component for the dashboard."""

import json
import os

import streamlit as st
from PIL import Image

from streamlit_app.config import PRIMARY_COLOR, BASE_DIR
from streamlit_app.data.loader import (
    load_latest_annex_iv,
    load_latest_release_info,
    load_manual_inputs,
    parse_requirements_txt,
    save_manual_inputs,
)
from streamlit_app.data.processor import render_markdown_with_newlines
from streamlit_app.utils.export import export_annex_iv_to_pdf


def display_annex_iv_documentation():
    """Display and allow editing of the Annex IV documentation."""
    st.markdown(
        '<div class="card">'
        "<h2>Annex IV Documentation</h2>"
        "<p>Technical documentation as required by EU AI Act Article 11.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load the latest Annex IV document and manual inputs
    annex_content, annex_path = load_latest_annex_iv()
    manual_inputs = load_manual_inputs()

    if annex_content is None:
        st.warning("No Annex IV documentation found.")
        return

    # Process the content to handle newlines properly and fix image paths
    processed_content = render_markdown_with_newlines(annex_content)

    # Tabs for viewing and editing
    tab1, tab2, tab3 = st.tabs(["📄 Documentation", "✏️ Edit Fields", "🔧 Framework Versions"])

    with tab1:
        # Display release info
        release_info = load_latest_release_info()
        if release_info:
            st.info(
                f"Showing documentation for release: **{release_info['id']}** (generated on {release_info['date']})"
            )

        # Add a document type/version badge at the top
        st.markdown(
            f'<div style="background-color: {PRIMARY_COLOR}; padding: 10px; '
            f'border-radius: 5px; color: white; display: inline-block; margin-bottom: 20px;">'
            f"Annex IV Technical Documentation</div>",
            unsafe_allow_html=True,
        )
        

        # Display the Annex IV document with proper newline handling
        st.markdown(processed_content)
        
        # Image sections that are known to have images in the document
        st.subheader("Document Images")
        st.info("These images are referenced in the document above")
        
        # Create columns for image display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Figure 1: System Architecture Overview**")
            e2e_path = os.path.join(BASE_DIR, "assets", "e2e.png")
            if os.path.exists(e2e_path):
                st.image(e2e_path, use_container_width=True)
            else:
                st.warning(f"Image not found: {e2e_path}")
                
        with col2:
            st.markdown("**Figure 2: Deployment Interface**")
            streamlit_path = os.path.join(BASE_DIR, "assets", "streamlit-app.png")
            if os.path.exists(streamlit_path):
                st.image(streamlit_path, use_container_width=True)
            else:
                st.warning(f"Image not found: {streamlit_path}")
        
        st.markdown("**Figure 3: Detailed System Architecture**")
        modal_path = os.path.join(BASE_DIR, "assets", "modal-deployment.png")
        if os.path.exists(modal_path):
            st.image(modal_path, use_container_width=True)
        else:
            st.warning(f"Image not found: {modal_path}")

        # Export options
        with st.expander("Export Options", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
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

            with col2:
                # Markdown export
                if st.button("Export as Markdown", key="annex_iv_md_export"):
                    st.download_button(
                        label="Download Markdown",
                        data=processed_content,
                        file_name="annex_iv_documentation.md",
                        mime="text/markdown",
                        key="download_annex_iv_md",
                    )

    with tab2:
        st.info(
            "Edit the fields below to update the Annex IV documentation. "
            "Changes will be applied the next time the Annex IV documentation is generated."
        )

        # Group inputs into logical sections with better styling
        with st.expander("General Information", expanded=True):
            manual_inputs["provider"] = st.text_input("Provider", manual_inputs.get("provider", ""))
            manual_inputs["description"] = st.text_area(
                "Description", manual_inputs.get("description", "")
            )
            manual_inputs["intended_purpose"] = st.text_area(
                "Intended Purpose", manual_inputs.get("intended_purpose", "")
            )

        with st.expander("System Architecture", expanded=False):
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

        with st.expander("Model Information", expanded=False):
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

        with st.expander("Risk and Compliance", expanded=False):
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

    with tab3:
        st.markdown("<h3>Framework Versions</h3>", unsafe_allow_html=True)
        st.info(
            "Framework versions are automatically loaded from requirements.txt. You can override them here if needed."
        )

        # Show current frameworks from requirements.txt
        current_frameworks = parse_requirements_txt()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4>From requirements.txt</h4>", unsafe_allow_html=True)
            if current_frameworks:
                framework_text = ""
                for package, version in current_frameworks.items():
                    framework_text += f"{package}: {version}\n"
                st.text_area(
                    "Current Framework Versions", framework_text, height=300, disabled=True
                )
            else:
                st.warning("No frameworks found in requirements.txt")

        with col2:
            st.markdown("<h4>Custom Framework Overrides</h4>", unsafe_allow_html=True)
            user_frameworks = manual_inputs.get("frameworks", {})

            # Use JSON editor for custom framework versions
            frameworks_json = json.dumps(user_frameworks, indent=2)
            frameworks_edited = st.text_area(
                "Custom Frameworks (JSON)", frameworks_json, height=300
            )
            try:
                manual_inputs["frameworks"] = json.loads(frameworks_edited)
            except json.JSONDecodeError:
                st.error("Invalid JSON format for frameworks")

            # Button to refresh from requirements.txt
            if st.button("Reset to requirements.txt", key="reset_frameworks"):
                manual_inputs["frameworks"] = parse_requirements_txt()
                st.success("Frameworks reset to values from requirements.txt")
                st.experimental_rerun()
