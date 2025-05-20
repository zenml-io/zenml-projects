"""Styling and CSS for the Streamlit dashboard."""

import streamlit as st

from streamlit_app.config import (
    ACCENT_COLOR,
    PRIMARY_COLOR,
    RISK_COLORS,
    SECONDARY_COLOR,
    TEXT_COLOR,
)


def apply_custom_css():
    """Add custom CSS to the app."""
    st.markdown(
        f"""
        <style>
        /* Custom theme and styling */
        :root {{
            --primary-color: {PRIMARY_COLOR};
            --secondary-color: {SECONDARY_COLOR};
            --accent-color: {ACCENT_COLOR};
            --text-color: {TEXT_COLOR};
        }}

        /* Main layout improvements */
        .main .block-container {{
            padding-top: 2rem;
        }}

        /* Card styles */
        .card {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}

        /* Header styling */
        h1, h2, h3 {{
            color: var(--primary-color);
            font-weight: 600;
        }}

        h1 {{
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}

        h2 {{
            margin-top: 1.5rem;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }}

        /* Metrics styling */
        .metric-card {{
            text-align: center;
            padding: 1rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }}

        .metric-label {{
            font-size: 1rem;
            color: var(--text-color);
        }}

        /* Table styling */
        .dataframe {{
            font-size: 0.9rem;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: #f0f2f6;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: var(--primary-color);
            color: white;
        }}

        /* Logo and branding */
        .logo-text {{
            font-size: 0.8rem;
            color: #666;
            margin-top: 0;
        }}

        .eu-compliance-badge {{
            display: inline-block;
            background-color: var(--primary-color);
            color: white;
            font-size: 0.7rem;
            padding: 5px 10px;
            border-radius: 20px;
            margin-bottom: 20px;
        }}

        /* Content dividers */
        .divider {{
            height: 3px;
            background-color: #f0f2f6;
            margin: 1rem 0;
        }}

        /* Tooltip improvement */
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: help;
        }}

        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}

        /* Navigation styling */
        .nav-link {{
            text-decoration: none;
            color: var(--text-color);
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 5px;
            transition: background-color 0.2s, color 0.2s;
        }}

        .nav-link:hover, .nav-link.active {{
            background-color: var(--primary-color);
            color: white;
        }}

        /* Iframe container */
        .iframe-container {{
            width: 100%;
            height: 800px;
            border: none;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        /* Last updated timestamp */
        .last-updated {{
            font-size: 0.8rem;
            color: #666;
            text-align: right;
            margin-top: 5px;
            font-style: italic;
        }}

        /* Status indicators */
        .status-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        
        .status-high {{
            background-color: {RISK_COLORS["HIGH"]}20;
            color: {RISK_COLORS["HIGH"]};
            border: 1px solid {RISK_COLORS["HIGH"]};
        }}
        
        .status-medium {{
            background-color: {RISK_COLORS["MEDIUM"]}20;
            color: {RISK_COLORS["MEDIUM"]};
            border: 1px solid {RISK_COLORS["MEDIUM"]};
        }}
        
        .status-low {{
            background-color: {RISK_COLORS["LOW"]}20;
            color: {RISK_COLORS["LOW"]};
            border: 1px solid {RISK_COLORS["LOW"]};
        }}
        
        /* Compliance score indicators */
        .compliance-score {{
            width: 70px;
            height: 70px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
            margin: 0 auto;
        }}
        
        .score-a {{
            background-color: #2E8B57;
        }}
        
        .score-b {{
            background-color: #3CB371;
        }}
        
        .score-c {{
            background-color: #FFB30F;
        }}
        
        .score-d {{
            background-color: #FF7F50;
        }}
        
        .score-f {{
            background-color: #D64045;
        }}

        /* Activity styling */
        .activity-container {{ 
            margin-top: 20px; 
        }}

        .activity-row {{
            display: flex;
            margin-bottom: 10px;
            padding: 12px 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            transition: box-shadow 0.3s, transform 0.3s;
            border-left: 3px solid #ccc;
        }}

        .activity-row:nth-child(odd) {{ 
            background-color: #ffffff; 
        }}

        .activity-row:hover {{
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
