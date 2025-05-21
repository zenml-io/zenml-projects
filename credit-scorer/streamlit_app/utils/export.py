"""Export utilities for the dashboard."""

import tempfile

import markdown
import streamlit as st
import weasyprint

from streamlit_app.config import PRIMARY_COLOR, SECONDARY_COLOR


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
                    color: {PRIMARY_COLOR};
                    border-bottom: 1px solid #eee;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: {SECONDARY_COLOR};
                    margin-top: 20px;
                }}
                h3 {{
                    color: {SECONDARY_COLOR};
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

        # Generate PDF using weasyprint
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

        try:
            html_doc = weasyprint.HTML(string=styled_html)
            html_doc.write_pdf(output_path)
            st.info("PDF generated using weasyprint.")
            return output_path
        except ImportError:
            st.error("weasyprint not installed. Install with: pip install weasyprint")
            raise
        except Exception as weasyprint_error:
            st.warning(f"weasyprint failed: {weasyprint_error}")

            # Final fallback - save as HTML
            html_output_path = output_path.replace(".pdf", ".html")
            with open(html_output_path, "w", encoding="utf-8") as f:
                f.write(styled_html)
            st.warning(f"Could not generate PDF. Saved as HTML instead: {html_output_path}")
            return html_output_path

    except Exception as e:
        st.error(f"Error exporting to PDF: {e}")
        return None
