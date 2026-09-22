from typing import List, Optional

import requests

from .config import CONTACT_EMAIL
from .models import Candidate

OPENALEX_URL = "https://api.openalex.org/works"

# Rotated monthly; add/adjust terms in sources.yaml eventually if this needs
# to grow past a hardcoded list.
QUERY_TERMS = [
    "facial recognition Canada policy",
    "facial recognition technology Canadian law",
    "biometric surveillance Canada regulation",
    "facial recognition police Canada",
]


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """OpenAlex stores abstracts as {word: [positions]} to dodge copyright
    reproduction of the original text — rebuild the plain string from it."""
    if not inverted_index:
        return None
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def _to_candidate(work: dict) -> Candidate:
    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
    authors = [
        a.get("author", {}).get("display_name")
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    url = primary_location.get("landing_page_url") or work.get("id")
    return Candidate(
        title=work.get("title") or "",
        url=url,
        source_family="academic",
        organization=source.get("display_name"),
        published_date=work.get("publication_date"),
        doi=doi,
        snippet=_reconstruct_abstract(work.get("abstract_inverted_index")),
        authors=authors,
        journal=source.get("display_name"),
    )


def _search_openalex(query: str, per_page: int = 25) -> List[dict]:
    params = {
        "search": query,
        "per-page": per_page,
        "filter": "from_publication_date:2015-01-01",
    }
    if CONTACT_EMAIL:
        # OpenAlex's "polite pool" — faster, more reliable rate limits. Opt-in
        # only (WATCH_CONTACT_EMAIL unset by default) since this gets sent to
        # OpenAlex's API — omitted rather than defaulted to anything.
        params["mailto"] = CONTACT_EMAIL
    resp = requests.get(OPENALEX_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def discover() -> List[Candidate]:
    candidates = []
    for query in QUERY_TERMS:
        for work in _search_openalex(query):
            candidates.append(_to_candidate(work))
    return candidates
