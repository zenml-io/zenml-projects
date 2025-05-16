from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Search:
    """Represents a search result from Tavily."""

    url: str = ""
    content: str = ""


@dataclass
class Research:
    """Container for search history and research progress for a paragraph."""

    search_history: List[Search] = field(default_factory=list)
    latest_summary: str = ""
    reflection_iteration: int = 0


@dataclass
class Paragraph:
    """Represents a paragraph in the research report."""

    title: str = ""
    content: str = ""
    research: Research = field(default_factory=Research)


@dataclass
class State:
    """The overall state of the research process."""

    report_title: str = ""
    query: str = ""  # Added to keep track of the original query
    paragraphs: List[Paragraph] = field(default_factory=list)
