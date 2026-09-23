import numpy as np
import faiss

import search_engine


def make_test_index():
    """
    Create a tiny deterministic FAISS index with 4 chunks.
    """
    vectors = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.0, 1.0],
    ], dtype=np.float32)

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    chunks = [
        {"text": "chunk A", "source": "source-a"},
        {"text": "chunk B", "source": "source-b"},
        {"text": "chunk C", "source": "source-c"},
        {"text": "chunk D", "source": "source-d"},
    ]

    return index, chunks


def test_search_retrieves_candidates_then_reranks(monkeypatch):
    index, chunks = make_test_index()

    # Make the query embedding deterministic.
    monkeypatch.setattr(
        search_engine,
        "embed",
        lambda texts: [[1.0, 0.0]],
    )

    captured = {}

    def fake_rerank(query, candidates):
        captured["query"] = query
        captured["candidates"] = candidates

        # Deliberately reverse the FAISS ordering.
        reranked = list(reversed(candidates))

        for i, result in enumerate(reranked):
            result["reranker_score"] = float(i)

        return reranked

    monkeypatch.setattr(search_engine, "rerank", fake_rerank)

    results = search_engine.search(
        index,
        chunks,
        "test query",
        top_k=2,
        candidate_k=3,
    )

    # Reranker should receive the requested candidate pool.
    assert captured["query"] == "test query"
    assert len(captured["candidates"]) == 3

    # FAISS metadata should survive reranking.
    assert captured["candidates"][0]["retrieval_rank"] == 1
    assert captured["candidates"][1]["retrieval_rank"] == 2
    assert captured["candidates"][2]["retrieval_rank"] == 3

    assert "retrieval_score" in captured["candidates"][0]

    # The reranker's ordering should determine the final output.
    assert results[0]["source"] == "source-c"
    assert results[1]["source"] == "source-b"

    # Only top_k final results should be returned.
    assert len(results) == 2

    # Final reranker ranks should be recorded.
    assert results[0]["reranker_rank"] == 1
    assert results[1]["reranker_rank"] == 2


def test_candidate_k_never_smaller_than_top_k(monkeypatch):
    index, chunks = make_test_index()

    monkeypatch.setattr(
        search_engine,
        "embed",
        lambda texts: [[1.0, 0.0]],
    )

    captured = {}

    def fake_rerank(query, candidates):
        captured["count"] = len(candidates)

        for candidate in candidates:
            candidate["reranker_score"] = 1.0

        return candidates

    monkeypatch.setattr(search_engine, "rerank", fake_rerank)

    results = search_engine.search(
        index,
        chunks,
        "test query",
        top_k=3,
        candidate_k=1,
    )

    # search() should automatically expand candidate_k to top_k.
    assert captured["count"] == 3
    assert len(results) == 3


def test_candidate_k_does_not_exceed_index_size(monkeypatch):
    index, chunks = make_test_index()

    monkeypatch.setattr(
        search_engine,
        "embed",
        lambda texts: [[1.0, 0.0]],
    )

    captured = {}

    def fake_rerank(query, candidates):
        captured["count"] = len(candidates)

        for candidate in candidates:
            candidate["reranker_score"] = 1.0

        return candidates

    monkeypatch.setattr(search_engine, "rerank", fake_rerank)

    search_engine.search(
        index,
        chunks,
        "test query",
        top_k=2,
        candidate_k=100,
    )

    # Tiny test index contains only 4 vectors.
    assert captured["count"] == 4


def test_empty_query_skips_reranker(monkeypatch):
    index, chunks = make_test_index()

    def should_not_run(*args, **kwargs):
        raise AssertionError("reranker should not run for an empty query")

    monkeypatch.setattr(search_engine, "rerank", should_not_run)

    assert search_engine.search(index, chunks, "") == []
    assert search_engine.search(index, chunks, "   ") == []