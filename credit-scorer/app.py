# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is a Flask web app that renders the Annex IV template and allows you to download it as a PDF."""

import os
import tempfile

import markdown
from flask import Flask, make_response, render_template, send_file
from jinja2 import Environment, FileSystemLoader
from weasyprint import CSS, HTML

app = Flask(__name__)
# Ensure templates directory exists and contains your template
app.jinja_env.add_extension("jinja2.ext.do")


def get_dummy_data():
    """Generate dummy data for the template."""
    return {
        "pipeline": {
            "name": "Credit Scoring Model v1.2",
            "version": "1.2.0",
            "description": "AI system for evaluating loan applications",
            "previous_versions": ["1.0.0", "1.1.0"],
            "steps": [
                {
                    "name": "Data preprocessing",
                    "type": "transform",
                    "description": "Clean and normalize input data",
                }
            ],
            "deployments": [{"type": "Docker container", "created_at": "2025-01-15"}],
        },
        "run": {
            "id": "run-123456",
            "code_reference": {"commit_sha": "abc123def456"},
            "metrics": {
                "accuracy": 0.92,
                "auc": 0.89,
                "fairness_metrics": {"disparity": 0.05, "demographic_parity": 0.97},
            },
            "metadata": {
                "model_type": "XGBoost ensemble",
                "optim_target": "F1 score with fairness constraints",
            },
        },
        "environment_variables": {"ORGANIZATION_NAME": "Example Corp."},
        "manual_inputs": {
            "design_assumptions": "System assumes complete financial history is available",
            "bias_mitigation": "Applied post-processing techniques to minimize gender and age bias",
        },
        # Add this to support the to_yaml filter
        "to_yaml": lambda x: yaml.dump(x, default_flow_style=False) if yaml else str(x),
    }


@app.route("/")
def preview():
    """Preview the template."""
    dummy_data = get_dummy_data()

    # First, render the template to get markdown content
    md_content = render_template("annex_iv_template.j2", **dummy_data)

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=["tables"])

    # Add some styling for better appearance
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Annex IV Documentation</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 30px; max-width: 1200px; }}
            h1, h2, h3, h4 {{ color: #333; margin-top: 1.5em; }}
            h1 {{ border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            table, th, td {{ border: 1px solid #ddd; }}
            th, td {{ padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            strong {{ color: #444; }}
            .download-options {{ margin-top: 50px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            .download-options ul {{ list-style-type: none; padding-left: 10px; }}
            .download-options li {{ margin: 10px 0; }}
            .download-options a {{ text-decoration: none; color: #0366d6; font-weight: bold; }}
            .download-options a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        {html_content}
        
        <div class="download-options">
            <h3>Download Options:</h3>
            <ul>
                <li>• <a href="/download/markdown">Download as Markdown</a></li>
                <li>• <a href="/download/pdf">Download as PDF</a></li>
            </ul>
        </div>
    </body>
    </html>
    """

    return styled_html


@app.route("/download/markdown")
def download_markdown():
    """Download the markdown version of the template."""
    dummy_data = get_dummy_data()

    # Render the template directly to get markdown content
    md_content = render_template("annex_iv_template.j2", **dummy_data)

    # Create a temporary file
    fd, path = tempfile.mkstemp(suffix=".md")
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write(md_content)

        # Send the file as a download
        return send_file(
            path,
            mimetype="text/markdown",
            as_attachment=True,
            download_name="annex_iv_documentation.md",
        )
    finally:
        # Ensure file is deleted after sending
        os.remove(path)


@app.route("/download/pdf")
def download_pdf():
    """Download the PDF version of the template."""
    dummy_data = get_dummy_data()

    # Render the template to get markdown content
    md_content = render_template("annex_iv_template.j2", **dummy_data)

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    # Add some basic CSS for better PDF formatting
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Annex IV Documentation</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 2cm; }}
            h1, h2, h3, h4 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            table, th, td {{ border: 1px solid #ddd; }}
            th, td {{ padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Create temporary files
    html_fd, html_path = tempfile.mkstemp(suffix=".html")
    pdf_fd, pdf_path = tempfile.mkstemp(suffix=".pdf")

    try:
        # Write HTML to temp file
        with os.fdopen(html_fd, "w") as tmp:
            tmp.write(styled_html)

        # Generate PDF
        HTML(html_path).write_pdf(pdf_path)

        # Send the PDF file as a download
        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="annex_iv_documentation.pdf",
        )
    finally:
        # Clean up temp files
        os.remove(html_path)
        os.remove(pdf_path)


if __name__ == "__main__":
    # Try to import yaml for to_yaml filter support
    try:
        import yaml
    except ImportError:
        yaml = None
        print("Warning: PyYAML not installed. to_yaml filter will have limited functionality.")

    app.run(debug=True)
