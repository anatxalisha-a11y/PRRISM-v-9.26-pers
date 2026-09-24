def test_hosted_reranker(monkeypatch):
    import retrieval.reranker as reranker

    candidates = [
        {"text": "first", "score": 0.9},
        {"text": "second", "score": 0.8},
        {"text": "third", "score": 0.7},
    ]

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"index": 2, "score": 4.0},
                {"index": 0, "score": 2.0},
                {"index": 1, "score": -1.0},
            ]

    def fake_post(url, headers, json, timeout):
        assert url == "https://example.test/rerank"

        assert headers["Authorization"] == "Bearer secret-token"

        assert json["query"] == "test query"
        assert json["texts"] == [
            "first",
            "second",
            "third",
        ]

        assert json["raw_scores"] is True
        assert json["return_text"] is False

        return FakeResponse()

    monkeypatch.setenv("RERANKER_BACKEND", "hosted")
    monkeypatch.setenv(
        "HF_RERANKER_ENDPOINT",
        "https://example.test",
    )
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    monkeypatch.setattr(
        reranker.requests,
        "post",
        fake_post,
    )

    results = reranker.rerank(
        "test query",
        candidates,
    )

    assert [r["text"] for r in results] == [
        "third",
        "first",
        "second",
    ]

    assert [r["reranker_score"] for r in results] == [
        4.0,
        2.0,
        -1.0,
    ]

    assert [r["retrieval_score"] for r in results] == [
        0.7,
        0.9,
        0.8,
    ]


def test_hosted_reranker_requires_endpoint(monkeypatch):
    import pytest
    import retrieval.reranker as reranker

    monkeypatch.setenv("RERANKER_BACKEND", "hosted")
    monkeypatch.delenv(
        "HF_RERANKER_ENDPOINT",
        raising=False,
    )
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    with pytest.raises(
        RuntimeError,
        match="HF_RERANKER_ENDPOINT",
    ):
        reranker.rerank(
            "query",
            [{"text": "document"}],
        )


def test_hosted_reranker_requires_token(monkeypatch):
    import pytest
    import retrieval.reranker as reranker

    monkeypatch.setenv("RERANKER_BACKEND", "hosted")
    monkeypatch.setenv(
        "HF_RERANKER_ENDPOINT",
        "https://example.test",
    )
    monkeypatch.delenv(
        "HF_TOKEN",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="HF_TOKEN",
    ):
        reranker.rerank(
            "query",
            [{"text": "document"}],
        )


def test_hosted_reranker_rejects_bad_result_count(monkeypatch):
    import pytest
    import retrieval.reranker as reranker

    candidates = [
        {"text": "first"},
        {"text": "second"},
    ]

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"index": 0, "score": 1.0},
            ]

    monkeypatch.setenv("RERANKER_BACKEND", "hosted")
    monkeypatch.setenv(
        "HF_RERANKER_ENDPOINT",
        "https://example.test",
    )
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    monkeypatch.setattr(
        reranker.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected number",
    ):
        reranker.rerank(
            "query",
            candidates,
        )


def test_invalid_reranker_backend(monkeypatch):
    import pytest
    import retrieval.reranker as reranker

    monkeypatch.setenv(
        "RERANKER_BACKEND",
        "something-weird",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported RERANKER_BACKEND",
    ):
        reranker.rerank(
            "query",
            [{"text": "document"}],
        )

def test_hosted_reranker_falls_back_on_request_failure(monkeypatch):
    import requests
    import retrieval.reranker as reranker

    candidates = [
        {"text": "first", "score": 0.9},
        {"text": "second", "score": 0.8},
        {"text": "third", "score": 0.7},
    ]

    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("endpoint unavailable")

    monkeypatch.setenv("RERANKER_BACKEND", "hosted")
    monkeypatch.setenv(
        "HF_RERANKER_ENDPOINT",
        "https://example.test",
    )
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    monkeypatch.setattr(
        reranker.requests,
        "post",
        fake_post,
    )

    results = reranker.rerank(
        "test query",
        candidates,
    )

    # Original FAISS ordering must be preserved.
    assert [r["text"] for r in results] == [
        "first",
        "second",
        "third",
    ]

    assert [r["retrieval_score"] for r in results] == [
        0.9,
        0.8,
        0.7,
    ]

    assert all(r["reranker_score"] is None for r in results)
    assert all(r["reranker_fallback"] is True for r in results)


def test_hosted_reranker_falls_back_on_http_429(monkeypatch):
    import requests
    import retrieval.reranker as reranker

    candidates = [
        {"text": "first", "score": 0.9},
        {"text": "second", "score": 0.8},
    ]

    class FakeResponse:
        status_code = 429

        def raise_for_status(self):
            raise requests.HTTPError("429 Too Many Requests")

    monkeypatch.setenv("RERANKER_BACKEND", "hosted")
    monkeypatch.setenv(
        "HF_RERANKER_ENDPOINT",
        "https://example.test",
    )
    monkeypatch.setenv("HF_TOKEN", "secret-token")

    monkeypatch.setattr(
        reranker.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    results = reranker.rerank(
        "test query",
        candidates,
    )

    assert [r["text"] for r in results] == [
        "first",
        "second",
    ]

    assert all(r["reranker_score"] is None for r in results)
    assert all(r["reranker_fallback"] is True for r in results)