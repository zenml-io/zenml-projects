import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing_extensions import Annotated
from zenml import step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


@step
def visualize_sales_data(
    sales_data: pd.DataFrame,
    train_data_dict: Dict[str, pd.DataFrame],
    test_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
) -> Annotated[HTMLString, "sales_visualization"]:
    """Create interactive visualizations of historical sales patterns.

    Args:
        sales_data: Raw sales data with date, store, item, and sales columns
        train_data_dict: Dictionary of training dataframes for each series
        test_data_dict: Dictionary of test dataframes for each series
        series_ids: List of unique series identifiers

    Returns:
        HTML visualization dashboard of historical sales patterns
    """
    # Ensure date column is in datetime format
    sales_data = sales_data.copy()
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Create HTML with multiple visualizations
    html_parts = []
    html_parts.append("""
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f9f9f9;
            }
            .dashboard {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .section {
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            h1, h2, h3 {
                color: #333;
            }
            .insights {
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-top: 10px;
            }
            .chart-container {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="section">
                <h1>Retail Sales Historical Data Analysis</h1>
                <p>Interactive visualization of sales patterns across stores and products.</p>
            </div>
    """)

    # Create overview metrics
    total_sales = sales_data["sales"].sum()
    avg_daily_sales = sales_data.groupby("date")["sales"].sum().mean()
    num_stores = sales_data["store"].nunique()
    num_items = sales_data["item"].nunique()
    min_date = sales_data["date"].min().strftime("%Y-%m-%d")
    max_date = sales_data["date"].max().strftime("%Y-%m-%d")
    date_range = f"{min_date} to {max_date}"

    html_parts.append(f"""
            <div class="section">
                <h2>Dataset Overview</h2>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                    <div style="flex: 1; min-width: 200px; background: #f0f8ff; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Total Sales</h3>
                        <p style="font-size: 24px; font-weight: bold;">{total_sales:,.0f} units</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #fff8f0; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Avg. Daily Sales</h3>
                        <p style="font-size: 24px; font-weight: bold;">{avg_daily_sales:,.1f} units</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #f0fff8; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Stores × Items</h3>
                        <p style="font-size: 24px; font-weight: bold;">{num_stores} × {num_items}</p>
                    </div>
                    <div style="flex: 1; min-width: 200px; background: #f8f0ff; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,0.1);">
                        <h3>Date Range</h3>
                        <p style="font-size: 18px; font-weight: bold;">{date_range}</p>
                    </div>
                </div>
            </div>
    """)

    # 1. Time Series - Overall Sales Trend
    df_daily = sales_data.groupby("date")["sales"].sum().reset_index()
    fig_trend = px.line(
        df_daily,
        x="date",
        y="sales",
        title="Daily Total Sales Across All Stores and Products",
        template="plotly_white",
    )
    fig_trend.update_traces(line=dict(width=2))
    fig_trend.update_layout(
        xaxis_title="Date", yaxis_title="Total Sales (units)", height=500
    )
    trend_html = fig_trend.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Overall Sales Trend</h2>
                {trend_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Observe weekly patterns and special events that impact overall sales volume.</p>
                </div>
            </div>
    """)

    # 2. Store Comparison
    store_sales = (
        sales_data.groupby(["date", "store"])["sales"].sum().reset_index()
    )
    fig_stores = px.line(
        store_sales,
        x="date",
        y="sales",
        color="store",
        title="Sales Comparison by Store",
        template="plotly_white",
    )
    fig_stores.update_layout(
        xaxis_title="Date", yaxis_title="Total Sales (units)", height=500
    )
    stores_html = fig_stores.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Store Comparison</h2>
                {stores_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Compare performance across different stores to identify top performers and potential issues.</p>
                </div>
            </div>
    """)

    # 3. Product Performance
    item_sales = (
        sales_data.groupby(["date", "item"])["sales"].sum().reset_index()
    )
    fig_items = px.line(
        item_sales,
        x="date",
        y="sales",
        color="item",
        title="Sales Comparison by Product",
        template="plotly_white",
    )
    fig_items.update_layout(
        xaxis_title="Date", yaxis_title="Total Sales (units)", height=500
    )
    items_html = fig_items.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Product Performance</h2>
                {items_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Identify best-selling products and those with unique seasonal patterns.</p>
                </div>
            </div>
    """)

    # 4. Weekly Patterns
    sales_data["day_of_week"] = sales_data["date"].dt.day_name()
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly_pattern = (
        sales_data.groupby("day_of_week")["sales"]
        .mean()
        .reindex(day_order)
        .reset_index()
    )

    fig_weekly = px.bar(
        weekly_pattern,
        x="day_of_week",
        y="sales",
        title="Average Sales by Day of Week",
        template="plotly_white",
        color="sales",
        color_continuous_scale="Blues",
    )
    fig_weekly.update_layout(
        xaxis_title="", yaxis_title="Average Sales (units)", height=500
    )
    weekly_html = fig_weekly.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Weekly Patterns</h2>
                {weekly_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Identify peak sales days to optimize inventory and staffing.</p>
                </div>
            </div>
    """)

    # 5. Sample Store-Item Combinations
    # Select 3 random series to display
    sample_series = np.random.choice(
        series_ids, size=min(3, len(series_ids)), replace=False
    )

    # Create subplots for train/test visualization
    fig_samples = make_subplots(
        rows=len(sample_series),
        cols=1,
        subplot_titles=[f"Series: {series_id}" for series_id in sample_series],
        shared_xaxes=True,
        vertical_spacing=0.1,
    )

    for i, series_id in enumerate(sample_series):
        train_data = train_data_dict[series_id]
        test_data = test_data_dict[series_id]

        # Add train data
        fig_samples.add_trace(
            go.Scatter(
                x=train_data["ds"],
                y=train_data["y"],
                mode="lines+markers",
                name=f"{series_id} (Training)",
                line=dict(color="blue"),
                legendgroup=series_id,
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

        # Add test data
        fig_samples.add_trace(
            go.Scatter(
                x=test_data["ds"],
                y=test_data["y"],
                mode="lines+markers",
                name=f"{series_id} (Test)",
                line=dict(color="green"),
                legendgroup=series_id,
                showlegend=(i == 0),
            ),
            row=i + 1,
            col=1,
        )

    fig_samples.update_layout(
        height=300 * len(sample_series),
        title_text="Train/Test Split for Sample Series",
        template="plotly_white",
    )

    samples_html = fig_samples.to_html(full_html=False, include_plotlyjs="cdn")
    html_parts.append(f"""
            <div class="section chart-container">
                <h2>Sample Series with Train/Test Split</h2>
                {samples_html}
                <div class="insights">
                    <p><strong>Insights:</strong> Visualize how historical data is split into training and testing sets for model evaluation.</p>
                </div>
            </div>
    """)

    # Close HTML document
    html_parts.append("""
        </div>
    </body>
    </html>
    """)

    # Combine all HTML parts
    complete_html = "".join(html_parts)

    # Return as HTMLString
    return HTMLString(complete_html)
