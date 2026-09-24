import pytest

from retrieval.diversity import select_diverse_results


def test_limits_chunks_per_source():
    results = [
        {"source": "NIST", "text": "NIST A"},
        {"source": "NIST", "text": "NIST B"},
        {"source": "NIST", "text": "NIST C"},
        {"source": "NIST", "text": "NIST D"},
        {"source": "AWS", "text": "AWS"},
        {"source": "Canada OPC", "text": "OPC"},
        {"source": "Clearview", "text": "Clearview"},
    ]

    selected = select_diverse_results(
        results,
        top_k=5,
        max_per_source=2,
    )

    assert [result["text"] for result in selected] == [
        "NIST A",
        "NIST B",
        "AWS",
        "OPC",
        "Clearview",
    ]


def test_backfills_when_not_enough_diverse_sources():
    results = [
        {"source": "NIST", "text": "NIST A"},
        {"source": "NIST", "text": "NIST B"},
        {"source": "NIST", "text": "NIST C"},
        {"source": "NIST", "text": "NIST D"},
        {"source": "AWS", "text": "AWS"},
    ]

    selected = select_diverse_results(
        results,
        top_k=5,
        max_per_source=2,
    )

    # First pass selects NIST A, NIST B, AWS.
    # Backfill restores NIST C and NIST D in relevance order.
    assert [result["text"] for result in selected] == [
        "NIST A",
        "NIST B",
        "AWS",
        "NIST C",
        "NIST D",
    ]


def test_preserves_order_when_results_are_already_diverse():
    results = [
        {"source": "A", "text": "A"},
        {"source": "B", "text": "B"},
        {"source": "C", "text": "C"},
        {"source": "D", "text": "D"},
        {"source": "E", "text": "E"},
    ]

    selected = select_diverse_results(
        results,
        top_k=5,
        max_per_source=2,
    )

    assert selected == results


def test_returns_only_top_k():
    results = [
        {"source": "A", "text": "A"},
        {"source": "B", "text": "B"},
        {"source": "C", "text": "C"},
        {"source": "D", "text": "D"},
    ]

    selected = select_diverse_results(
        results,
        top_k=2,
        max_per_source=2,
    )

    assert len(selected) == 2
    assert selected[0]["text"] == "A"
    assert selected[1]["text"] == "B"


def test_empty_results():
    assert select_diverse_results([]) == []


def test_zero_top_k():
    results = [{"source": "A", "text": "A"}]

    assert select_diverse_results(
        results,
        top_k=0,
    ) == []


def test_invalid_max_per_source():
    results = [{"source": "A", "text": "A"}]

    with pytest.raises(ValueError):
        select_diverse_results(
            results,
            max_per_source=0,
        )

def test_source_id_is_canonical_source_identifier():
    results = [
        {
            "source_id": "ABC123",
            "title": "Same document",
            "chunk_id": 0,
        },
        {
            "source_id": "ABC123",
            "title": "Same document",
            "chunk_id": 1,
        },
        {
            "source_id": "ABC123",
            "title": "Same document",
            "chunk_id": 2,
        },
        {
            "source_id": "XYZ789",
            "title": "Another document",
            "chunk_id": 0,
        },
    ]

    selected = select_diverse_results(
        results,
        top_k=3,
        max_per_source=2,
    )

    assert selected[0]["chunk_id"] == 0
    assert selected[0]["source_id"] == "ABC123"

    assert selected[1]["chunk_id"] == 1
    assert selected[1]["source_id"] == "ABC123"

    assert selected[2]["source_id"] == "XYZ789"