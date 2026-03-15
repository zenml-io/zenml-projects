"""Data models for the email search agent environment."""

from dataclasses import dataclass
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Email(BaseModel):
    """Represents an email from the Enron dataset."""

    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = Field(default_factory=list)
    cc_addresses: List[str] = Field(default_factory=list)
    bcc_addresses: List[str] = Field(default_factory=list)
    body: Optional[str] = None
    file_name: Optional[str] = None


class Scenario(BaseModel):
    """A question-answer scenario for training/evaluation."""

    id: int
    question: str
    answer: str
    message_ids: List[str]  # message_ids of referenced emails
    how_realistic: float
    inbox_address: str
    query_date: str
    split: Literal["train", "test"]


@dataclass
class SearchResult:
    """Result from searching the email database."""

    message_id: str
    snippet: str


class FinalAnswer(BaseModel):
    """The agent's final answer with source references."""

    answer: str
    source_ids: List[str]


class EmailScenario(BaseModel):
    """Wrapper for scenario with training step info."""

    step: int
    scenario: Scenario
