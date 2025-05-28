"""CSS utility functions for consistent styling across materializers."""

import json
import os
from typing import Optional


def get_shared_css_path() -> str:
    """Get the absolute path to the shared CSS file.

    Returns:
        Absolute path to assets/styles.css
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "assets", "styles.css")


def get_shared_css_content() -> str:
    """Read and return the content of the shared CSS file.

    Returns:
        Content of the shared CSS file
    """
    css_path = get_shared_css_path()
    try:
        with open(css_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to basic styles if file not found
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            color: #333;
        }
        """


def get_shared_css_tag() -> str:
    """Get the complete style tag with shared CSS content.

    Returns:
        HTML style tag with shared CSS
    """
    css_content = get_shared_css_content()
    return f"<style>\n{css_content}\n</style>"


def get_confidence_class(level: str) -> str:
    """Return appropriate CSS class for confidence level.

    Args:
        level: Confidence level (high, medium, low)

    Returns:
        CSS class string
    """
    return f"dr-confidence dr-confidence--{level.lower()}"


def get_badge_class(badge_type: str) -> str:
    """Return appropriate CSS class for badges.

    Args:
        badge_type: Badge type (success, warning, danger, info, primary)

    Returns:
        CSS class string
    """
    return f"dr-badge dr-badge--{badge_type.lower()}"


def get_status_class(status: str) -> str:
    """Return appropriate CSS class for status indicators.

    Args:
        status: Status type (approved, pending, rejected, etc.)

    Returns:
        CSS class string
    """
    status_map = {
        "approved": "success",
        "pending": "warning",
        "rejected": "danger",
        "completed": "success",
        "in_progress": "info",
        "failed": "danger",
    }
    badge_type = status_map.get(status.lower(), "primary")
    return get_badge_class(badge_type)


def get_section_class(section_type: Optional[str] = None) -> str:
    """Return appropriate CSS class for sections.

    Args:
        section_type: Optional section type (info, warning, success, danger)

    Returns:
        CSS class string
    """
    if section_type:
        return f"dr-section dr-section--{section_type.lower()}"
    return "dr-section"


def get_card_class(hoverable: bool = True) -> str:
    """Return appropriate CSS class for cards.

    Args:
        hoverable: Whether the card should have hover effects

    Returns:
        CSS class string
    """
    classes = ["dr-card"]
    if not hoverable:
        classes.append("dr-card--no-hover")
    return " ".join(classes)


def get_table_class(striped: bool = False) -> str:
    """Return appropriate CSS class for tables.

    Args:
        striped: Whether the table should have striped rows

    Returns:
        CSS class string
    """
    classes = ["dr-table"]
    if striped:
        classes.append("dr-table--striped")
    return " ".join(classes)


def get_button_class(
    button_type: str = "primary", size: str = "normal"
) -> str:
    """Return appropriate CSS class for buttons.

    Args:
        button_type: Button type (primary, secondary, success)
        size: Button size (normal, small)

    Returns:
        CSS class string
    """
    classes = ["dr-button"]
    if button_type != "primary":
        classes.append(f"dr-button--{button_type}")
    if size == "small":
        classes.append("dr-button--small")
    return " ".join(classes)


def get_grid_class(grid_type: str = "cards") -> str:
    """Return appropriate CSS class for grid layouts.

    Args:
        grid_type: Grid type (stats, cards, metrics)

    Returns:
        CSS class string
    """
    return f"dr-grid dr-grid--{grid_type}"


def wrap_with_container(content: str, wide: bool = False) -> str:
    """Wrap content with container div.

    Args:
        content: HTML content to wrap
        wide: Whether to use wide container

    Returns:
        Wrapped HTML content
    """
    container_class = (
        "dr-container dr-container--wide" if wide else "dr-container"
    )
    return f'<div class="{container_class}">{content}</div>'


def create_stat_card(value: str, label: str, format_value: bool = True) -> str:
    """Create a stat card HTML.

    Args:
        value: The statistic value
        label: The label for the statistic
        format_value: Whether to wrap value in stat-value div

    Returns:
        HTML for stat card
    """
    value_html = (
        f'<div class="dr-stat-value">{value}</div>' if format_value else value
    )
    return f"""
    <div class="dr-stat-card">
        {value_html}
        <div class="dr-stat-label">{label}</div>
    </div>
    """


def create_notice(content: str, notice_type: str = "info") -> str:
    """Create a notice box HTML.

    Args:
        content: Notice content
        notice_type: Notice type (info, warning)

    Returns:
        HTML for notice box
    """
    return f"""
    <div class="dr-notice dr-notice--{notice_type}">
        {content}
    </div>
    """


def extract_html_from_content(content: str) -> str:
    """Attempt to extract HTML content from a response that might be wrapped in other formats.

    Args:
        content: The content to extract HTML from

    Returns:
        The extracted HTML, or a basic fallback if extraction fails
    """
    if not content:
        return ""

    # Try to find HTML between tags
    if "<html" in content and "</html>" in content:
        start = content.find("<html")
        end = content.find("</html>") + 7  # Include the closing tag
        return content[start:end]

    # Try to find div class="research-report"
    if '<div class="research-report"' in content and "</div>" in content:
        start = content.find('<div class="research-report"')
        # Find the last closing div
        last_div = content.rfind("</div>")
        if last_div > start:
            return content[start : last_div + 6]  # Include the closing tag

    # Look for code blocks
    if "```html" in content and "```" in content:
        start = content.find("```html") + 7
        end = content.find("```", start)
        if end > start:
            return content[start:end].strip()

    # Look for JSON with an "html" field
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "html" in parsed:
            return parsed["html"]
    except:
        pass

    # If all extraction attempts fail, return the original content
    return content
