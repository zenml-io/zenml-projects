"""Data Profile component for the dashboard."""

import streamlit as st

from streamlit_app.data.loader import (
    load_latest_release_info,
    load_latest_whylogs_profile,
)


def display_data_profile():
    """Display the whylogs data profile."""
    st.markdown(
        '<div class="card">'
        "<h2>Data Profile</h2>"
        "<p>WhyLogs data profiling report required by EU AI Act Article 10 (Data Governance).</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Get the latest whylogs profile
    profile_path = load_latest_whylogs_profile()

    if profile_path is None:
        st.warning("No whylogs profile found in the latest release.")
        return

    # Display the profile in an iframe
    release_info = load_latest_release_info()
    if release_info:
        st.markdown(
            f"Showing data profile for release: **{release_info['id']}** (generated on {release_info['date']})"
        )

    # Create an iframe to display the HTML content
    html_file_path = str(profile_path)
    with open(html_file_path, "r") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=800, scrolling=True)
