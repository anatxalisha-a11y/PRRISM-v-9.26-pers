import os
from typing import List, Dict

from sentence_transformers import CrossEncoder


RERANK_MODEL_NAME = os.environ.get(
    "RERANK_MODEL_NAME",
    "BAAI/bge-reranker-v2-m3",
)

# Load once when this module is imported.
_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)

    return _reranker


def rerank(
    query: str,
    candidates: List[Dict],
) -> List[Dict]:
    """
    Rerank candidate chunks against a query using a cross-encoder.

    Each candidate must contain a `text` field.

    Returns a new list ordered from most to least relevant.
    The original FAISS retrieval score is preserved as
    `retrieval_score`, and the cross-encoder score is stored as
    `reranker_score`.
    """
    if not query or not query.strip() or not candidates:
        return []

    pairs = [
        (query.strip(), candidate["text"])
        for candidate in candidates
    ]

    scores = get_reranker().predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = []

    for candidate, reranker_score in zip(candidates, scores):
        result = dict(candidate)

        # Preserve the original FAISS score explicitly.
        if "retrieval_score" not in result and "score" in result:
            result["retrieval_score"] = result["score"]

        result["reranker_score"] = float(reranker_score)
        reranked.append(result)

    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked