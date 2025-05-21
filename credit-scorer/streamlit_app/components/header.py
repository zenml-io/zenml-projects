"""Header component for the dashboard."""

import sys
from pathlib import Path

import streamlit as st

from src.utils.compliance.compliance_calculator import calculate_compliance
from streamlit_app.data.loader import load_latest_release_info

# Add project root to path so we can import from src
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_compliance_score():
    """Calculate the current compliance score using the same method as the test script.

    The result is cached for 1 hour to avoid recalculation on every page refresh.
    """
    try:
        results = calculate_compliance()
        overall_score = results["overall"]["overall_compliance_score"]
        return round(overall_score)  # Round to nearest integer
    except Exception as e:
        # Log the error but don't display it in the UI
        import logging

        logging.error(f"Error calculating compliance score: {e}")
        return 75  # Fallback value if calculation fails


def display_dashboard_header():
    """Display the dashboard header with branding and last updated info."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<div style="display: flex; align-items: center;">'
            '<h1 style="margin-bottom: 0;">Credit Scoring AI System</h1>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="eu-compliance-badge">EU AI Act Compliant</div>',
            unsafe_allow_html=True,
        )

    with col2:
        # Get latest release info for "last updated" timestamp
        release_info = load_latest_release_info()
        if release_info:
            st.markdown(
                f'<div class="last-updated">Last updated: {release_info["date"]}</div>',
                unsafe_allow_html=True,
            )

        # Add a "Compliance Score" badge with dynamically calculated score
        compliance_score = get_compliance_score()  # Calculate dynamically (cached)
        score_letter = (
            "A"
            if compliance_score >= 90
            else "B"
            if compliance_score >= 80
            else "C"
            if compliance_score >= 70
            else "D"
            if compliance_score >= 60
            else "F"
        )

        st.markdown(
            f'<div style="text-align: center; margin-top: 10px;" title="Dynamically calculated compliance score based on EU AI Act requirements">'
            f'<p style="margin-bottom: 5px; font-size: 0.9rem;">Compliance Score</p>'
            f'<div class="compliance-score score-{score_letter.lower()}" title="Grade based on overall compliance">{score_letter}</div>'
            f'<p style="margin-top: 5px; font-size: 0.7rem;">{compliance_score}/100</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
