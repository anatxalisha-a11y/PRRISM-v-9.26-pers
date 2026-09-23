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
    min_sources: int | None = None,
) -> List[Dict]:
    """
    Select highly ranked results while limiting repeated chunks
    from the same source.

    When min_sources is not supplied, preserve the original behavior:
    return up to top_k chunks while respecting max_per_source where
    possible.

    When min_sources is supplied, first ensure that up to that many
    unique sources are represented. Additional highly ranked chunks
    may then be included up to max_per_source for those sources.

    Input results must already be ordered by relevance.
    """

    if not results or top_k <= 0:
        return []

    if max_per_source <= 0:
        raise ValueError(
            "max_per_source must be greater than 0"
        )

    if min_sources is not None and min_sources <= 0:
        raise ValueError(
            "min_sources must be greater than 0"
        )

    # ------------------------------------------------------------------
    # Original chunk-count behavior
    # ------------------------------------------------------------------

    if min_sources is None:

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

        # If the diversity constraint leaves us short, restore the
        # strongest skipped chunks in original relevance order.
        for result in skipped:

            selected.append(result)

            if len(selected) == top_k:
                break

        return selected

    # ------------------------------------------------------------------
    # Source-count behavior
    # ------------------------------------------------------------------

    target_sources = min_sources

    selected = []
    selected_ids = set()
    seen_sources = set()

    # First pass:
    # Take the highest-ranked chunk from each new source until the
    # requested number of unique sources has been represented.
    for result in results:

        source = _source_key(result)

        if source in seen_sources:
            continue

        selected.append(result)
        selected_ids.add(id(result))
        seen_sources.add(source)

        if len(seen_sources) >= target_sources:
            break

    # There may simply not be enough unique sources in the candidate
    # pool. In that case, return everything we were able to represent.
    actual_sources = len(seen_sources)

    if actual_sources == 0:
        return []

    # We want enough evidence to represent the requested sources while
    # still allowing multiple useful passages from a source.
    #
    # top_k acts as the normal chunk budget, but it can never be smaller
    # than the number of represented sources.
    chunk_budget = max(
        top_k,
        actual_sources,
    )

    source_counts = defaultdict(int)

    for result in selected:
        source_counts[
            _source_key(result)
        ] += 1

    # Second pass:
    # Revisit the relevance-ranked list and add useful additional chunks
    # from sources that made the source-level selection.
    for result in results:

        if len(selected) >= chunk_budget:
            break

        if id(result) in selected_ids:
            continue

        source = _source_key(result)

        if source not in seen_sources:
            continue

        if source_counts[source] >= max_per_source:
            continue

        selected.append(result)
        selected_ids.add(id(result))
        source_counts[source] += 1

    # Restore original relevance order after source coverage selection.
    selected_id_set = {
        id(result)
        for result in selected
    }

    return [
        result
        for result in results
        if id(result) in selected_id_set
    ]