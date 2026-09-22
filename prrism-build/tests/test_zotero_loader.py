import sys
from pathlib import Path

import pytest


BUILD_DIR = Path(__file__).resolve().parents[1]

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

from zotero_loader import (
    APPROVED_COLLECTION_NAMES,
    load_approved_sources,
    resolve_approved_collections,
    walk_collection_tree,
)


class FakeZotero:
    """
    Small in-memory representation of Zotero's collection tree.

    subcollections maps:
        parent collection key -> direct child collections

    items maps:
        collection key -> direct top-level items
    """

    def __init__(
        self,
        subcollections=None,
        items=None,
    ):
        self.subcollections = subcollections or {}
        self.items = items or {}

    def get_subcollections(self, collection_key):
        return self.subcollections.get(
            collection_key,
            [],
        )

    def get_collection_items(self, collection_key):
        value = self.items.get(
            collection_key,
            [],
        )

        if isinstance(value, Exception):
            raise value

        return value


def collection(key, name):
    return {
        "key": key,
        "data": {
            "key": key,
            "name": name,
        },
    }


def item(
    key,
    url,
    title,
    item_type="webpage",
    content_type="",
    link_mode="",
    filename="",
):
    return {
        "key": key,
        "data": {
            "key": key,
            "url": url,
            "title": title,
            "itemType": item_type,
            "contentType": content_type,
            "linkMode": link_mode,
            "filename": filename,
        },
    }


@pytest.fixture
def approved_collections():
    return [
        collection(
            "BENCH123",
            "Benchmarks",
        ),
        collection(
            "BEST1234",
            "Best Practices",
        ),
        collection(
            "POLICY12",
            "Policy",
        ),
        collection(
            "CAND1234",
            "Candidate Sources",
        ),
    ]


@pytest.fixture
def approved_tree(approved_collections):
    return {
        "ROOT": approved_collections,
    }


def test_resolves_only_approved_collections(
    approved_collections,
):
    zotero = FakeZotero(
        subcollections={
            "ROOT": approved_collections,
        }
    )

    resolved = resolve_approved_collections(
        zotero,
        "ROOT",
    )

    assert set(resolved) == APPROVED_COLLECTION_NAMES
    assert "Candidate Sources" not in resolved


def test_missing_approved_collection_fails():
    zotero = FakeZotero(
        subcollections={
            "ROOT": [
                collection(
                    "BENCH123",
                    "Benchmarks",
                ),
                collection(
                    "POLICY12",
                    "Policy",
                ),
            ],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Best Practices",
    ):
        resolve_approved_collections(
            zotero,
            "ROOT",
        )


def test_loads_sources_from_all_approved_collections(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "BENCH123": [
                item(
                    "ITEM0001",
                    "https://example.com/a",
                    "A",
                ),
            ],
            "BEST1234": [
                item(
                    "ITEM0002",
                    "https://example.com/b",
                    "B",
                ),
            ],
            "POLICY12": [
                item(
                    "ITEM0003",
                    "https://example.com/c",
                    "C",
                ),
            ],

            # Must never be read by the loader.
            "CAND1234": [
                item(
                    "CAND0001",
                    "https://example.com/candidate",
                    "Candidate",
                ),
            ],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert [
        source.source_id
        for source in sources
    ] == [
        "ITEM0001",
        "ITEM0002",
        "ITEM0003",
    ]


def test_recursively_loads_descendant_collections(
    approved_collections,
):
    """
    The important new behaviour:

    Best Practices
        -> Amazon
        -> Microsoft

    Policy
        -> CAN
            -> Privacy Commissioner

    All descendant items should enter the approved corpus.
    """
    amazon = collection(
        "AMAZON01",
        "Amazon (Rekognition)",
    )
    microsoft = collection(
        "MICRO001",
        "Microsoft (Azure Face ID)",
    )
    canada = collection(
        "CAN00001",
        "CAN",
    )
    privacy = collection(
        "PRIV0001",
        "Office of the Privacy Commissioner of Canada",
    )

    zotero = FakeZotero(
        subcollections={
            "ROOT": approved_collections,
            "BEST1234": [
                amazon,
                microsoft,
            ],
            "POLICY12": [
                canada,
            ],
            "CAN00001": [
                privacy,
            ],
        },
        items={
            "AMAZON01": [
                item(
                    "AMZITEM1",
                    "https://example.com/amazon",
                    "Amazon Source",
                ),
            ],
            "MICRO001": [
                item(
                    "MSITEM01",
                    "https://example.com/microsoft",
                    "Microsoft Source",
                ),
            ],
            "PRIV0001": [
                item(
                    "CANITEM1",
                    "https://example.com/privacy",
                    "Canadian Privacy Source",
                ),
            ],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert {
        source.source_id
        for source in sources
    } == {
        "AMZITEM1",
        "MSITEM01",
        "CANITEM1",
    }

    by_id = {
        source.source_id: source
        for source in sources
    }

    assert by_id[
        "AMZITEM1"
    ].collections == (
        "Amazon (Rekognition)",
    )

    assert by_id[
        "MSITEM01"
    ].collections == (
        "Microsoft (Azure Face ID)",
    )

    assert by_id[
        "CANITEM1"
    ].collections == (
        "Office of the Privacy Commissioner of Canada",
    )


def test_walk_collection_tree_includes_root_and_descendants():
    child = collection(
        "CHILD001",
        "Child",
    )
    grandchild = collection(
        "GRAND001",
        "Grandchild",
    )

    zotero = FakeZotero(
        subcollections={
            "ROOT1234": [
                child,
            ],
            "CHILD001": [
                grandchild,
            ],
        }
    )

    result = list(
        walk_collection_tree(
            zotero,
            collection_key="ROOT1234",
            collection_name="Root",
        )
    )

    assert result == [
        (
            "ROOT1234",
            "Root",
        ),
        (
            "CHILD001",
            "Child",
        ),
        (
            "GRAND001",
            "Grandchild",
        ),
    ]


def test_candidate_sources_and_descendants_are_excluded(
    approved_collections,
):
    candidate_child = collection(
        "CANDCHLD",
        "Candidate Child",
    )

    zotero = FakeZotero(
        subcollections={
            "ROOT": approved_collections,
            "CAND1234": [
                candidate_child,
            ],
        },
        items={
            "CAND1234": [
                item(
                    "CAND0001",
                    "https://example.com/candidate",
                    "Candidate",
                ),
            ],
            "CANDCHLD": [
                item(
                    "CAND0002",
                    "https://example.com/candidate-child",
                    "Candidate Child Source",
                ),
            ],

            # Keep the approved corpus non-empty.
            "POLICY12": [
                item(
                    "GOOD0001",
                    "https://example.com/policy",
                    "Approved Policy",
                ),
            ],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert [
        source.source_id
        for source in sources
    ] == [
        "GOOD0001",
    ]


def test_duplicate_memberships_are_merged(
    approved_tree,
):
    shared = item(
        "SHARED01",
        "https://example.com/shared",
        "Shared Source",
    )

    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "BENCH123": [
                shared,
            ],
            "BEST1234": [],
            "POLICY12": [
                shared,
            ],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert len(sources) == 1
    assert sources[0].source_id == "SHARED01"

    assert sources[0].collections == (
        "Benchmarks",
        "Policy",
    )


def test_duplicate_descendant_memberships_are_merged(
    approved_collections,
):
    amazon = collection(
        "AMAZON01",
        "Amazon (Rekognition)",
    )
    policy_child = collection(
        "POLCHILD",
        "CAN",
    )

    shared = item(
        "SHARED01",
        "https://example.com/shared",
        "Shared Source",
    )

    zotero = FakeZotero(
        subcollections={
            "ROOT": approved_collections,
            "BEST1234": [
                amazon,
            ],
            "POLICY12": [
                policy_child,
            ],
        },
        items={
            "AMAZON01": [
                shared,
            ],
            "POLCHILD": [
                shared,
            ],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert len(sources) == 1

    assert sources[0].collections == (
        "Amazon (Rekognition)",
        "CAN",
    )


def test_missing_url_is_skipped(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "BENCH123": [
                item(
                    "GOOD0001",
                    "https://example.com",
                    "Good",
                ),
                item(
                    "BAD00001",
                    "",
                    "No URL",
                ),
            ],
            "BEST1234": [],
            "POLICY12": [],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert len(sources) == 1
    assert sources[0].source_id == "GOOD0001"


def test_stored_zotero_file_without_url_is_included(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "POLICY12": [
                item(
                    "PDF00001",
                    "",
                    "Stored PDF",
                    item_type="attachment",
                    content_type="application/pdf",
                    link_mode="imported_file",
                    filename="stored.pdf",
                ),
            ],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert len(sources) == 1

    source = sources[0]

    assert source.source_id == "PDF00001"
    assert source.url == ""
    assert source.content_type == "application/pdf"
    assert source.link_mode == "imported_file"
    assert source.attachment_key == "PDF00001"


def test_missing_title_falls_back_to_url(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "BENCH123": [
                item(
                    "ITEM0001",
                    "https://example.com/source",
                    "",
                ),
            ],
            "BEST1234": [],
            "POLICY12": [],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert (
        sources[0].title
        == "https://example.com/source"
    )


def test_stored_file_title_can_fall_back_to_filename(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "POLICY12": [
                item(
                    "PDF00001",
                    "",
                    "",
                    item_type="attachment",
                    content_type="application/pdf",
                    link_mode="imported_file",
                    filename="report.pdf",
                ),
            ],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert sources[0].title == "report.pdf"


def test_results_are_deterministic(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "BENCH123": [
                item(
                    "ZZZZZZZZ",
                    "https://example.com/z",
                    "Z",
                ),
                item(
                    "AAAAAAAA",
                    "https://example.com/a",
                    "A",
                ),
            ],
            "BEST1234": [],
            "POLICY12": [],
        },
    )

    sources = load_approved_sources(
        zotero,
        "ROOT",
    )

    assert [
        source.source_id
        for source in sources
    ] == [
        "AAAAAAAA",
        "ZZZZZZZZ",
    ]


def test_empty_approved_corpus_fails(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "BENCH123": [],
            "BEST1234": [],
            "POLICY12": [],
        },
    )

    with pytest.raises(
        RuntimeError,
        match="No approved PRRISM sources",
    ):
        load_approved_sources(
            zotero,
            "ROOT",
        )


def test_zotero_api_failure_is_not_treated_as_empty(
    approved_tree,
):
    zotero = FakeZotero(
        subcollections=approved_tree,
        items={
            "BENCH123": RuntimeError(
                "Zotero unavailable"
            ),
            "BEST1234": [],
            "POLICY12": [],
        },
    )

    with pytest.raises(
        RuntimeError,
        match="Zotero unavailable",
    ):
        load_approved_sources(
            zotero,
            "ROOT",
        )