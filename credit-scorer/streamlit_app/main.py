"""Main entry point for the Streamlit dashboard application."""

import streamlit as st
from streamlit_option_menu import option_menu

from streamlit_app.components.api_dashboard import display_api_dashboard
from streamlit_app.components.data_profile import display_data_profile
from streamlit_app.components.documentation import (
    display_annex_iv_documentation,
)
from streamlit_app.components.executive_summary import display_exec_summary
from streamlit_app.components.header import display_dashboard_header
from streamlit_app.components.incidents import display_incident_tracking
from streamlit_app.components.risks import display_risks_dashboard
from streamlit_app.data.loader import load_incident_log, load_risk_register
from streamlit_app.styling import apply_custom_css


def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="Credit Scoring AI - EU AI Act Compliance Dashboard",
        page_icon="📊",
        layout="wide",
    )

    # Apply custom CSS
    apply_custom_css()

    # Display dashboard header
    display_dashboard_header()

    # Navigation using a more modern approach
    selected = option_menu(
        menu_title=None,
        options=[
            "Executive Summary",
            "Risk Management",
            "Incident Tracking",
            "API Dashboard",
            "Data Profile",
            "Annex IV Documentation",
        ],
        icons=[
            "clipboard-data",
            "shield-exclamation",
            "exclamation-triangle",
            "cloud",
            "bar-chart",
            "file-earmark-text",
        ],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0px",
                "background-color": "#f8f9fa",
                "border-radius": "10px",
                "margin-bottom": "20px",
            },
            "icon": {"color": "#1F4E79", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#1F4E79"},
        },
    )

    # Load shared data
    risk_data = load_risk_register()
    incident_data = load_incident_log()

    # Display the selected page
    if selected == "Executive Summary":
        risk_df = (
            risk_data.get("Risks", pd.DataFrame())
            if risk_data
            else pd.DataFrame()
        )
        display_exec_summary(risk_df, incident_data)

    elif selected == "Risk Management":
        if risk_data and "Risks" in risk_data:
            display_risks_dashboard(risk_data["Risks"])
        else:
            st.warning("Risk register not found or empty.")

    elif selected == "Incident Tracking":
        display_incident_tracking(incident_data)

    elif selected == "API Dashboard":
        display_api_dashboard()

    elif selected == "Data Profile":
        display_data_profile()

    elif selected == "Annex IV Documentation":
        display_annex_iv_documentation()

    # Footer with credits
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; padding: 20px; font-size: 0.8rem; color: #666;">
            <p>Credit Scoring AI System - EU AI Act Compliance Dashboard</p>
            <p>© 2025 ZenML GmbH. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    import pandas as pd

    main()
