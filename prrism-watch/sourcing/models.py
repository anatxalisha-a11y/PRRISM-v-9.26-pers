from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Candidate:
    """A single discovered page/article, moving through filter -> dedup -> Zotero insert."""

    title: str
    url: str
    source_family: str  # "government" | "municipal" | "company" | "news" | "academic"

    organization: Optional[str] = None
    published_date: Optional[str] = None
    doi: Optional[str] = None
    snippet: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    journal: Optional[str] = None

    # filled in by filter_candidates.judge_candidate
    relevant: Optional[bool] = None
    reason: Optional[str] = None
    practice_tags: List[str] = field(default_factory=list)
    ethics_tags: List[str] = field(default_factory=list)
    jurisdiction_tags: List[str] = field(default_factory=list)
