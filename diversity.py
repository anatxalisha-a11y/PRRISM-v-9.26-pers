from collections import defaultdict
from typing import Dict, List


def _source_key(result: Dict) -> str:
    """
    Return the canonical document identifier used for source diversity.

    PRRISM chunks use `source_id` as the Zotero item key, so all chunks
    originating from the same approved source share the same source_id.
    """
    source_id = result.get("source_id")

    if source_id:
        return str(source_id)

    # Defensive fallback for results that predate the current chunk schema.
    return str(
        result.get("url")
        or result.get("title")
        or result.get("source")
        or result.get("chunk_id")
        or id(result)
    )

def select_diverse_results(
    results: List[Dict],
    top_k: int = 5,
    max_per_source: int = 2,
) -> List[Dict]:
    """
    Select top results while limiting repeated chunks from the same source.

    The input is assumed to already be ordered by relevance.

    First pass:
        Respect max_per_source.

    Second pass:
        If fewer than top_k results were selected, backfill from skipped
        candidates in their original relevance order.
    """
    if not results or top_k <= 0:
        return []

    if max_per_source <= 0:
        raise ValueError("max_per_source must be greater than 0")

    selected = []
    skipped = []
    source_counts = defaultdict(int)

    for result in results:
        source = _source_key(result)

        if source_counts[source] < max_per_source:
            selected.append(result)
            source_counts[source] += 1

            if len(selected) == top_k:
                return selected
        else:
            skipped.append(result)

    # If diversity constraints left us short, restore the strongest
    # skipped results in their original relevance order.
    for result in skipped:
        selected.append(result)

        if len(selected) == top_k:
            break

    return selected