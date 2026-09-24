import pytest

from answer_generator import (
    build_citation_map,
    build_evidence,
    validate_generated_answer,
)


# =============================================================================
# SAMPLE DATA
# =============================================================================

def _sample_evidence():
    return [
        {
            "citation_id": "S1",
            "source_id": "DOC-A",
            "title": "Source A",
            "url": "https://example.com/a",
            "collections": ["Policy"],
            "themes": {
                "ethics": ["fairness"],
                "practice": ["human oversight"],
                "jurisdiction": ["Canada"],
            },
            "passages": [
                "Evidence A.",
            ],
        },
        {
            "citation_id": "S2",
            "source_id": "DOC-B",
            "title": "Source B",
            "url": "https://example.com/b",
            "collections": ["Research"],
            "themes": {
                "ethics": [],
                "practice": [],
                "jurisdiction": [],
            },
            "passages": [
                "Evidence B.",
            ],
        },
    ]


def _sample_generated_results():
    return {
        "results": [
            {
                "citation_id": "S1",
                "summary": (
                    "Source A discusses privacy concerns "
                    "associated with facial recognition."
                ),
            },
            {
                "citation_id": "S2",
                "summary": (
                    "Source B discusses oversight in the "
                    "use of facial recognition."
                ),
            },
        ]
    }


# =============================================================================
# BUILD EVIDENCE
# =============================================================================

def test_build_evidence_groups_chunks_by_source():
    results = [
        {
            "source_id": "DOC-A",
            "title": "Source A",
            "url": "https://example.com/a",
            "collections": ["Policy"],
            "text": "First passage from A.",
            "ethics_tags": [
                {
                    "tag": "fairness",
                    "score": 0.81,
                },
                {
                    "tag": "ethics",
                    "score": 0.72,
                },
            ],
            "practice_tags": [
                {
                    "tag": "human oversight",
                    "score": 0.75,
                },
            ],
            "jurisdiction_tags": [
                {
                    "tag": "Canada",
                    "score": 0.90,
                },
            ],
        },
        {
            "source_id": "DOC-B",
            "title": "Source B",
            "url": "https://example.com/b",
            "collections": ["Research"],
            "text": "Passage from B.",
            "ethics_tags": [],
            "practice_tags": [],
            "jurisdiction_tags": [],
        },
        {
            "source_id": "DOC-A",
            "title": "Source A",
            "url": "https://example.com/a",
            "collections": ["Policy"],
            "text": "Second passage from A.",
            "ethics_tags": [
                {
                    "tag": "risk framework",
                    "score": 0.66,
                },
            ],
            "practice_tags": [
                {
                    "tag": "accountability",
                    "score": 0.61,
                },
            ],
            "jurisdiction_tags": [
                {
                    "tag": "Canada",
                    "score": 0.84,
                },
            ],
        },
    ]

    evidence = build_evidence(
        results
    )

    assert len(evidence) == 2

    source_a = evidence[0]
    source_b = evidence[1]

    assert source_a[
        "citation_id"
    ] == "S1"

    assert source_a[
        "source_id"
    ] == "DOC-A"

    assert source_a[
        "title"
    ] == "Source A"

    assert source_a[
        "url"
    ] == "https://example.com/a"

    assert source_a[
        "collections"
    ] == ["Policy"]

    assert source_a[
        "passages"
    ] == [
        "First passage from A.",
        "Second passage from A.",
    ]

    assert source_a[
        "themes"
    ] == {
        "ethics": [
            "fairness",
            "ethics",
            "risk framework",
        ],
        "practice": [
            "human oversight",
            "accountability",
        ],
        "jurisdiction": [
            "Canada",
        ],
    }

    assert source_b[
        "citation_id"
    ] == "S2"

    assert source_b[
        "source_id"
    ] == "DOC-B"

    assert source_b[
        "passages"
    ] == [
        "Passage from B.",
    ]


def test_build_evidence_hides_internal_scores():
    results = [
        {
            "source_id": "DOC-A",
            "title": "Source A",
            "text": "Evidence.",
            "score": 0.91,
            "retrieval_score": 0.91,
            "reranker_score": 4.2,
            "retrieval_rank": 3,
            "reranker_rank": 1,
            "ethics_tags": [
                {
                    "tag": "fairness",
                    "score": 0.88,
                },
            ],
        }
    ]

    evidence = build_evidence(
        results
    )

    source = evidence[0]

    assert "score" not in source
    assert "retrieval_score" not in source
    assert "reranker_score" not in source
    assert "retrieval_rank" not in source
    assert "reranker_rank" not in source

    assert source[
        "passages"
    ] == [
        "Evidence.",
    ]

    assert source[
        "themes"
    ][
        "ethics"
    ] == [
        "fairness",
    ]


def test_build_evidence_preserves_ranked_source_order():
    results = [
        {
            "source_id": "B",
            "title": "B",
            "text": "B passage",
        },
        {
            "source_id": "A",
            "title": "A",
            "text": "A passage",
        },
        {
            "source_id": "B",
            "title": "B",
            "text": "Another B passage",
        },
    ]

    evidence = build_evidence(
        results
    )

    assert [
        source["source_id"]
        for source in evidence
    ] == [
        "B",
        "A",
    ]

    assert [
        source["citation_id"]
        for source in evidence
    ] == [
        "S1",
        "S2",
    ]

    assert evidence[0][
        "passages"
    ] == [
        "B passage",
        "Another B passage",
    ]


def test_build_evidence_rejects_missing_source_id():
    with pytest.raises(
        ValueError,
        match="missing source_id",
    ):
        build_evidence(
            [
                {
                    "title": "Source",
                    "text": "Evidence.",
                }
            ]
        )


# =============================================================================
# VALIDATE RANKED RESULTS
# =============================================================================

def test_validate_generated_answer_builds_ranked_results():
    result = (
        _sample_generated_results()
    )

    validated = (
        validate_generated_answer(
            result,
            _sample_evidence(),
        )
    )

    assert validated == {
        "results": [
            {
                "rank": 1,
                "citation_id": "S1",
                "source_id": "DOC-A",
                "title": "Source A",
                "url": (
                    "https://example.com/a"
                ),
                "summary": (
                    "Source A discusses privacy "
                    "concerns associated with "
                    "facial recognition."
                ),
            },
            {
                "rank": 2,
                "citation_id": "S2",
                "source_id": "DOC-B",
                "title": "Source B",
                "url": (
                    "https://example.com/b"
                ),
                "summary": (
                    "Source B discusses oversight "
                    "in the use of facial "
                    "recognition."
                ),
            },
        ]
    }


def test_validate_generated_answer_preserves_prrism_rank():
    result = (
        _sample_generated_results()
    )

    validated = (
        validate_generated_answer(
            result,
            _sample_evidence(),
        )
    )

    assert [
        item["rank"]
        for item in validated[
            "results"
        ]
    ] == [
        1,
        2,
    ]

    assert [
        item["citation_id"]
        for item in validated[
            "results"
        ]
    ] == [
        "S1",
        "S2",
    ]

    assert [
        item["source_id"]
        for item in validated[
            "results"
        ]
    ] == [
        "DOC-A",
        "DOC-B",
    ]


def test_validate_generated_answer_rejects_reordered_sources():
    result = {
        "results": [
            {
                "citation_id": "S2",
                "summary": (
                    "Summary for source B."
                ),
            },
            {
                "citation_id": "S1",
                "summary": (
                    "Summary for source A."
                ),
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="do not match PRRISM ranking",
    ):
        validate_generated_answer(
            result,
            _sample_evidence(),
        )


def test_validate_generated_answer_rejects_unknown_source():
    result = {
        "results": [
            {
                "citation_id": "S1",
                "summary": (
                    "Summary for source A."
                ),
            },
            {
                "citation_id": "S99",
                "summary": (
                    "Invented source."
                ),
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="do not match PRRISM ranking",
    ):
        validate_generated_answer(
            result,
            _sample_evidence(),
        )


def test_validate_generated_answer_rejects_missing_source():
    result = {
        "results": [
            {
                "citation_id": "S1",
                "summary": (
                    "Summary for source A."
                ),
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="count does not match",
    ):
        validate_generated_answer(
            result,
            _sample_evidence(),
        )


def test_validate_generated_answer_rejects_extra_source():
    result = {
        "results": [
            {
                "citation_id": "S1",
                "summary": (
                    "Summary for source A."
                ),
            },
            {
                "citation_id": "S2",
                "summary": (
                    "Summary for source B."
                ),
            },
            {
                "citation_id": "S3",
                "summary": (
                    "Unexpected source."
                ),
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="count does not match",
    ):
        validate_generated_answer(
            result,
            _sample_evidence(),
        )


def test_validate_generated_answer_rejects_empty_summary():
    result = {
        "results": [
            {
                "citation_id": "S1",
                "summary": "   ",
            },
            {
                "citation_id": "S2",
                "summary": (
                    "Summary for source B."
                ),
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="Summary must not be empty",
    ):
        validate_generated_answer(
            result,
            _sample_evidence(),
        )


def test_validate_generated_answer_rejects_generated_metadata():
    result = {
        "results": [
            {
                "citation_id": "S1",
                "summary": (
                    "Summary for source A."
                ),
                "title": (
                    "Model-generated title"
                ),
            },
            {
                "citation_id": "S2",
                "summary": (
                    "Summary for source B."
                ),
            },
        ]
    }

    with pytest.raises(
        ValueError,
        match="invalid fields",
    ):
        validate_generated_answer(
            result,
            _sample_evidence(),
        )


def test_validate_generated_answer_rejects_extra_top_level_fields():
    result = {
        "results": [
            {
                "citation_id": "S1",
                "summary": (
                    "Summary for source A."
                ),
            },
            {
                "citation_id": "S2",
                "summary": (
                    "Summary for source B."
                ),
            },
        ],
        "answer": (
            "The model should not generate "
            "a combined answer."
        ),
    }

    with pytest.raises(
        ValueError,
        match="invalid fields",
    ):
        validate_generated_answer(
            result,
            _sample_evidence(),
        )


# =============================================================================
# TRUSTED SOURCE METADATA
# =============================================================================

def test_build_citation_map_uses_trusted_evidence_metadata():
    citation_map = (
        build_citation_map(
            _sample_evidence()
        )
    )

    assert citation_map == {
        "S1": {
            "source_id": "DOC-A",
            "title": "Source A",
            "url": (
                "https://example.com/a"
            ),
        },
        "S2": {
            "source_id": "DOC-B",
            "title": "Source B",
            "url": (
                "https://example.com/b"
            ),
        },
    }


def test_build_citation_map_preserves_missing_url():
    evidence = [
        {
            "citation_id": "S1",
            "source_id": "DOC-A",
            "title": "Source A",
            "url": None,
            "collections": [],
            "themes": {},
            "passages": [
                "Evidence.",
            ],
        }
    ]

    citation_map = (
        build_citation_map(
            evidence
        )
    )

    assert (
        citation_map[
            "S1"
        ][
            "url"
        ]
        is None
    )


def test_build_citation_map_does_not_expose_passages_or_themes():
    citation_map = (
        build_citation_map(
            _sample_evidence()
        )
    )

    for citation in (
        citation_map.values()
    ):
        assert (
            "passages"
            not in citation
        )

        assert (
            "themes"
            not in citation
        )

        assert (
            "collections"
            not in citation
        )