from typing import List
from urllib.parse import urlsplit

from .models import Candidate
from .search_client import CREDITS_PER_QUERY, search_web

# News is deliberately left on the cheapest path (a generic search API) rather
# than a curated list — but the query set below is written to favor primary
# sources (official releases, regulator/court text, company statements) over
# secondary commentary, and filter_candidates' LLM judge does a second pass
# for source quality.
DEFAULT_QUERIES = [
    'facial recognition Canada "news release"',
    "facial recognition technology Canada court ruling",
    "facial recognition ban Canada police",
    "facial recognition Canada privacy commissioner investigation",
]

# Tavily charges 1 credit per basic search regardless of max_results, so
# credits used this run == number of queries run. A monthly cron with the 4
# default queries costs 4 credits/month — nothing close to a free-trial
# budget — but this caps it defensively in case sources.yaml's news_queries
# list ever grows without someone doing that math again.
MAX_QUERIES_PER_RUN = 50


def discover(sources_yaml: dict) -> List[Candidate]:
    queries = sources_yaml.get("news_queries") or DEFAULT_QUERIES
    if len(queries) > MAX_QUERIES_PER_RUN:
        print(
            f"[discover_news] news_queries has {len(queries)} entries, "
            f"truncating to {MAX_QUERIES_PER_RUN} to cap Tavily credit spend this run"
        )
        queries = queries[:MAX_QUERIES_PER_RUN]

    candidates = []
    credits_used = 0
    for query in queries:
        try:
            results = search_web(query)
            credits_used += CREDITS_PER_QUERY
        except Exception as e:  # noqa: BLE001 - a missing/broken search API shouldn't take down gov/municipal/company discovery
            print(f"[discover_news] search failed for {query!r}: {e}")
            continue
        for item in results:
            url = item.get("url", "")
            candidates.append(Candidate(
                title=item.get("title", ""),
                url=url,
                source_family="news",
                organization=urlsplit(url).netloc or None,
                snippet=item.get("content"),
            ))

    print(f"[discover_news] used ~{credits_used} Tavily credits this run ({len(queries)} queries x {CREDITS_PER_QUERY} credit/query)")
    return candidates
