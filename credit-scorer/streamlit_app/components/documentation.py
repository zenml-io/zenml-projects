"""Annex IV Documentation component for the dashboard."""

import re

import streamlit as st

from streamlit_app.config import PRIMARY_COLOR
from streamlit_app.data.loader import (
    load_latest_annex_iv,
    load_latest_release_info,
)
from streamlit_app.data.processor import render_markdown_with_newlines
from streamlit_app.utils.export import export_annex_iv_to_pdf


def display_annex_iv_documentation():
    """Display the Annex IV documentation with export options."""
    st.markdown(
        '<div class="card">'
        "<h2>Annex IV Documentation</h2>"
        "<p>Technical documentation as required by EU AI Act Article 11.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load the latest Annex IV document
    annex_content, annex_path = load_latest_annex_iv()

    if annex_content is None:
        st.warning("No Annex IV documentation found.")
        return

    # Process the content to handle newlines properly and fix image paths
    processed_content = render_markdown_with_newlines(annex_content)

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

    # Remove images from content for display (they may not render properly in Streamlit)
    content_without_images = re.sub(
        r"!\[.*?\]\(.*?assets/.*?\)\n*", "", processed_content
    )

    # Display the documentation content
    st.markdown(content_without_images)

    # Export options
    st.markdown("---")
    st.subheader("Export Options")

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
