import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from whylogs.core import DatasetProfileView
from zenml.logger import get_logger
from zenml.types import HTMLString

logger = get_logger(__name__)


def _format_num(val: Any, precision: int = 6) -> str:
    """Convert a numeric value to string, trim trailing zeros & dots."""
    try:
        f = float(val)
    except Exception:
        return str(val)
    # preserve nan
    if pd.isna(f):
        return "N/A"
    # format with fixed precision, then strip
    s = format(f, f".{precision}f").rstrip("0").rstrip(".")
    return s


def generate_whylogs_visualization(
    dataset_info: Dict[str, Any],
    data_profile: DatasetProfileView,
) -> HTMLString:
    """Generate HTML visualization for WhyLogs profile data.

    Args:
        dataset_info: Dataset information
        data_profile: WhyLogs profile view

    Returns:
        HTMLString containing the visualization
    """
    # Convert profile to pandas DataFrame for better inspection
    profile_df = data_profile.to_pandas()

    # Start building HTML content
    html_content = """
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .profile-summary {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .data-summary {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .alert {{ background-color: #fff3cd; padding: 15px; border-left: 6px solid #ffc107; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Credit Scoring Dataset Profile</h1>
        
        <div class="data-summary">
            <h2>Dataset Summary</h2>
            <ul>
                <li><strong>Rows:</strong> {rows}</li>
                <li><strong>Columns:</strong> {columns}</li>
                <li><strong>Missing Values:</strong> {missing}</li>
                <li><strong>Data Source:</strong> {source}</li>
            </ul>
        </div>
    """.format(
        rows=dataset_info["rows"],
        columns=dataset_info["columns"],
        missing=dataset_info["missing_values"],
        source=os.path.basename(dataset_info["source"]),
    )

    # Create the main statistics table
    html_content += """
        <div class="profile-summary">
            <h2>Column Statistics</h2>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Count</th>
                    <th>Null Count</th>
                    <th>Unique Count</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                </tr>
    """

    # Add rows for each column in the profile
    for col in profile_df.index:
        try:
            row = profile_df.loc[col]

            # Find the right metric names based on what's available
            metrics = row.index
            count_metric = next((m for m in metrics if m in ("counts/n", "distribution/n")), None)
            null_metric = next((m for m in metrics if m in ("counts/null", "counts/nan")), None)
            unique_metric = next((m for m in metrics if m.startswith("cardinality/")), None)
            min_metric = next((m for m in metrics if m.endswith("/min")), None)
            max_metric = next((m for m in metrics if m.endswith("/max")), None)
            mean_metric = next((m for m in metrics if m.endswith("/mean")), None)

            # Get values with error handling
            count_val = row[count_metric] if count_metric else "N/A"
            null_val = row[null_metric] if null_metric else "N/A"

            # Format the values below
            unique_val = _format_num(row[unique_metric], 0) if unique_metric else "N/A"
            min_val = _format_num(row[min_metric]) if min_metric else "N/A"
            max_val = _format_num(row[max_metric]) if max_metric else "N/A"
            mean_val = _format_num(row[mean_metric], 4) if mean_metric else "N/A"

            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{count_val}</td>
                    <td>{null_val}</td>
                    <td>{unique_val}</td>
                    <td>{min_val}</td>
                    <td>{max_val}</td>
                    <td>{mean_val}</td>
                </tr>
            """
        except Exception:
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td colspan="7">No statistics available</td>
                </tr>
            """

    html_content += """
            </table>
        </div>
    """

    # Add section about sensitive attributes if they exist
    if "sensitive_attributes" in dataset_info and dataset_info["sensitive_attributes"]:
        html_content += """
            <div class="alert">
                <h3>Sensitive Attributes Detected</h3>
                <p>The following columns contain potentially sensitive information:</p>
                <ul>
        """
        for col in dataset_info["sensitive_attributes"]:
            html_content += f"<li>{col}</li>"

        html_content += """
                </ul>
                <p>These attributes should be handled with care in compliance with the EU AI Act.</p>
            </div>
        """

    # Close HTML
    html_content += """
    </body>
    </html>
    """

    # Save HTML file
    output_dir = os.path.join(os.getcwd(), "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"whylogs_profile_{timestamp}.html"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w") as f:
        f.write(html_content)

    logger.info(f"WhyLogs visualization saved to: {os.path.abspath(file_path)}")

    return HTMLString(html_content)
