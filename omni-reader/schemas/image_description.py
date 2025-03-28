"""This module contains the schema for the OCR results."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ImageDescription(BaseModel):
    """Base model for OCR results."""

    raw_text: str = Field(description="Extracted text from the image")
    description: str = Field(description="Description of the image")
    entities: Optional[List[str]] = Field(default=None, description="List of entities found in the image")
