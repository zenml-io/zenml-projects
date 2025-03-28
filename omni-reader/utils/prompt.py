"""This module contains the prompt for the OCR model."""

from typing import Optional


def get_prompt(
    custom_prompt: Optional[str] = None,
) -> str:
    """Get the prompt for the OCR model.

    Args:
        custom_prompt: A custom prompt for the OCR model.

    Returns:
        str: The prompt for the OCR model.
    """
    if custom_prompt:
        return custom_prompt
    return (
        "First, describe the image in detail. "
        "Then, extract raw text from the image. "
        "Finally, list any entities present "
        "(e.g., fictional characters, objects, locations, etc.)."
    )
