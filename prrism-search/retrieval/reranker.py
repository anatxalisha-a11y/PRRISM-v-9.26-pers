"""Reranking backends for PRRISM retrieval.

Production can use the hosted Hugging Face TEI endpoint.
Local inference remains available for development and evaluation.
"""

import os
from typing import Dict, List

import requests
from sentence_transformers import CrossEncoder

import logging


RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DEFAULT_BACKEND = "local"
REQUEST_TIMEOUT = 30

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    """Lazy-load the local cross-encoder."""
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)

    return _reranker


def _prepare_candidates(candidates: List[Dict]) -> List[Dict]:
    """Copy candidates and preserve the original retrieval score."""
    prepared = []

    for candidate in candidates:
        item = dict(candidate)

        if "retrieval_score" not in item and "score" in item:
            item["retrieval_score"] = item["score"]

        prepared.append(item)

    return prepared


def _rerank_local(query: str, candidates: List[Dict]) -> List[Dict]:
    """Rerank candidates with the local sentence-transformers model."""
    pairs = [(query, candidate["text"]) for candidate in candidates]

    scores = get_reranker().predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["reranker_score"] = float(score)

    return sorted(
        candidates,
        key=lambda item: item["reranker_score"],
        reverse=True,
    )


def _rerank_hosted(query: str, candidates: List[Dict]) -> List[Dict]:
    """Rerank candidates using a private Hugging Face TEI endpoint."""
    endpoint = os.environ.get("HF_RERANKER_ENDPOINT")
    token = os.environ.get("HF_TOKEN")

    if not endpoint:
        raise RuntimeError(
            "HF_RERANKER_ENDPOINT is required when "
            "RERANKER_BACKEND=hosted."
        )

    if not token:
        raise RuntimeError(
            "HF_TOKEN is required when RERANKER_BACKEND=hosted."
        )

    url = f"{endpoint.rstrip('/')}/rerank"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "texts": [candidate["text"] for candidate in candidates],
            "raw_scores": True,
            "return_text": False,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    results = response.json()

    if len(results) != len(candidates):
        raise RuntimeError(
            "Hosted reranker returned an unexpected number of results."
        )

    ranked = []

    for result in results:
        index = result.get("index")
        score = result.get("score")

        if not isinstance(index, int):
            raise RuntimeError(
                "Hosted reranker returned an invalid candidate index."
            )

        if index < 0 or index >= len(candidates):
            raise RuntimeError(
                "Hosted reranker returned an out-of-range candidate index."
            )

        if score is None:
            raise RuntimeError(
                "Hosted reranker returned a result without a score."
            )

        candidate = dict(candidates[index])
        candidate["reranker_score"] = float(score)
        ranked.append(candidate)

    return ranked

def _fallback_to_retrieval(
    candidates: List[Dict],
    error: Exception,
) -> List[Dict]:
    """Preserve FAISS ordering when the hosted reranker is unavailable."""
    logger.warning(
        "Hosted reranker unavailable; falling back to retrieval order: %s",
        error,
    )

    fallback = []

    for candidate in candidates:
        item = dict(candidate)
        item["reranker_score"] = None
        item["reranker_fallback"] = True
        fallback.append(item)

    return fallback

def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    """Rerank retrieval candidates using the configured backend."""
    if not candidates:
        return []

    prepared = _prepare_candidates(candidates)

    backend = os.environ.get(
        "RERANKER_BACKEND",
        DEFAULT_BACKEND,
    ).strip().lower()

    if backend == "local":
        return _rerank_local(query, prepared)

    if backend == "hosted":
        # Missing credentials/configuration are deployment errors.
        # Do not silently hide them with the retrieval fallback.
        endpoint = os.environ.get("HF_RERANKER_ENDPOINT")
        token = os.environ.get("HF_TOKEN")

        if not endpoint:
            raise RuntimeError(
                "HF_RERANKER_ENDPOINT is required when "
                "RERANKER_BACKEND=hosted."
            )

        if not token:
            raise RuntimeError(
                "HF_TOKEN is required when "
                "RERANKER_BACKEND=hosted."
            )

        try:
            return _rerank_hosted(query, prepared)
        except requests.RequestException as error:
            return _fallback_to_retrieval(
                prepared,
                error,
            )

    # Anything other than the explicitly supported backends
    # is a configuration error.
    raise ValueError(
        f"Unsupported RERANKER_BACKEND: {backend!r}. "
        "Expected 'local' or 'hosted'."
    )