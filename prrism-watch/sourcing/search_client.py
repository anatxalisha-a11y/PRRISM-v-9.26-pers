from typing import List

import requests

from .config import TAVILY_API_KEY

SEARCH_URL = "https://api.tavily.com/search"

# Tavily bills per search call, not per result: 1 credit for search_depth
# "basic" (what this pipeline always uses), 2 for "advanced". Keep this in
# sync with the search_depth value below — discover_news.py imports it to
# report actual spend per run.
CREDITS_PER_QUERY = 1


def search_web(query: str, max_results: int = 10) -> List[dict]:
    """Returns a list of {title, url, content} dicts from Tavily."""
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY not set — see prrism-watch/README.md.")
    resp = requests.post(
        SEARCH_URL,
        headers={
            "Authorization": f"Bearer {TAVILY_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"query": query, "max_results": max_results, "search_depth": "basic"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])
