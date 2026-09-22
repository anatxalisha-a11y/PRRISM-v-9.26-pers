"""Pure unit tests — no network, no API keys. Run with:
    cd prrism-watch && python -m pytest tests/ -q
"""
import json
import os
import sys
from unittest.mock import patch

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sourcing.filter_candidates as filter_candidates_module  # noqa: E402
from sourcing.dedup import (  # noqa: E402
    dedup_key,
    normalize_url,
    sync_zotero_seen_state,
    zotero_seen_keys,
)
from sourcing.filter_candidates import filter_candidates, keyword_prefilter_pass  # noqa: E402
from sourcing.item_builder import ITEM_TYPE_BY_FAMILY, build_zotero_item  # noqa: E402
from sourcing.models import Candidate  # noqa: E402
from sourcing.run_pipeline import load_sources  # noqa: E402
from sourcing.tags import canada_terms, load_taxonomy  # noqa: E402


class FakeGroqResponse:
    """Stands in for requests.Response for _post_to_groq tests."""

    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"{self.status_code} error")
            err.response = self
            raise err

    def json(self):
        return self._json_data


def _verdict_response(verdict: dict) -> FakeGroqResponse:
    return FakeGroqResponse(200, {
        "choices": [{"message": {"content": json.dumps(verdict)}}]
    })


class FakeZoteroClient:
    """Stands in for ZoteroClient in sync tests — no network. tree carries
    the collection-key set the "tree" resolves to; items_by_version is a
    list of (items_batch, resulting_version) tuples returned in order, one
    per call to get_items_since, to simulate successive incremental syncs."""

    def __init__(self, tree: set, items_by_version: list):
        self.tree = tree
        self._batches = list(items_by_version)

    def get_collection_tree_keys(self, root_collection_key):
        return self.tree

    def get_items_since(self, since_version):
        return self._batches.pop(0)


def test_normalize_url_strips_www_trailing_slash_and_tracking_params():
    a = normalize_url("https://WWW.Example.com/Path/?utm_source=x&b=2&a=1")
    b = normalize_url("https://example.com/Path?a=1&b=2")
    assert a == b


def test_dedup_key_prefers_doi_over_url():
    c = Candidate(title="t", url="https://example.com/a", source_family="academic", doi="10.1/ABC")
    assert dedup_key(c) == "doi:10.1/abc"


def test_dedup_key_falls_back_to_normalized_url():
    c = Candidate(title="t", url="https://example.com/a", source_family="news")
    assert dedup_key(c) == f"url:{normalize_url('https://example.com/a')}"


def test_keyword_prefilter_requires_fr_term():
    c = Candidate(title="Unrelated municipal budget update", url="https://x.ca", source_family="government")
    assert keyword_prefilter_pass(c) is False


def test_keyword_prefilter_curated_families_skip_canada_term_requirement():
    c = Candidate(title="New facial recognition guidance", url="https://x.ca", source_family="government")
    assert keyword_prefilter_pass(c) is True


def test_keyword_prefilter_news_requires_both_fr_and_canada_terms():
    fr_only = Candidate(title="Facial recognition rollout announced", url="https://x.com", source_family="news")
    fr_and_ca = Candidate(title="Facial recognition rollout announced in Ontario", url="https://x.com", source_family="news")
    assert keyword_prefilter_pass(fr_only) is False
    assert keyword_prefilter_pass(fr_and_ca) is True


def test_taxonomy_loads_and_has_canada_jurisdiction():
    taxonomy = load_taxonomy()
    assert "Canada" in taxonomy["jurisdiction_tags"]
    assert "Ontario" in canada_terms()


def test_item_builder_maps_source_family_to_zotero_item_type():
    assert ITEM_TYPE_BY_FAMILY["academic"] == "journalArticle"
    assert ITEM_TYPE_BY_FAMILY["government"] == "webpage"


def test_build_zotero_item_sets_target_collection_and_tags():
    c = Candidate(
        title="Guidance on FR use",
        url="https://priv.gc.ca/example",
        source_family="government",
        organization="Office of the Privacy Commissioner of Canada",
        practice_tags=["regulatory compliance"],
        ethics_tags=["laws and regulations"],
        jurisdiction_tags=["Canada"],
    )
    template = {"itemType": "webpage", "title": "", "url": "", "websiteTitle": "", "tags": [], "collections": []}
    item = build_zotero_item(c, template)
    assert item["collections"] == ["CSZS3XH5"]
    tag_values = {t["tag"] for t in item["tags"]}
    assert "source-family:government" in tag_values
    assert "practice:regulatory compliance" in tag_values
    assert "jurisdiction:Canada" in tag_values


def test_sources_yaml_is_valid_and_loadable():
    sources = load_sources()
    assert isinstance(sources, dict)


def test_sync_zotero_seen_state_first_run_indexes_items_in_tree_only():
    tree = {"ROOT1", "SUB1"}
    items = [
        {"data": {"key": "A", "url": "https://x.ca/a", "collections": ["ROOT1"]}},
        {"data": {"key": "B", "url": "https://x.ca/b", "collections": ["SUB1"]}},
        {"data": {"key": "C", "url": "https://x.ca/c", "collections": ["OTHER"]}},  # outside the tree
    ]
    client = FakeZoteroClient(tree=tree, items_by_version=[(items, 10)])

    state = sync_zotero_seen_state(client, "ROOT1", {"version": 0, "items": {}})

    assert state["version"] == 10
    assert set(state["items"].keys()) == {"A", "B"}
    seen = zotero_seen_keys(state)
    assert f"url:{normalize_url('https://x.ca/a')}" in seen
    assert f"url:{normalize_url('https://x.ca/c')}" not in seen


def test_sync_zotero_seen_state_incremental_run_only_processes_delta():
    tree = {"ROOT1"}
    seeded_state = {"version": 10, "items": {"A": {"url": "https://x.ca/a", "doi": None}}}
    delta = [{"data": {"key": "D", "url": "https://x.ca/d", "collections": ["ROOT1"]}}]
    client = FakeZoteroClient(tree=tree, items_by_version=[(delta, 11)])

    state = sync_zotero_seen_state(client, "ROOT1", seeded_state)

    assert state["version"] == 11
    assert set(state["items"].keys()) == {"A", "D"}  # A carried forward, D added


def test_sync_zotero_seen_state_drops_items_moved_out_of_tree():
    tree = {"ROOT1"}
    seeded_state = {"version": 10, "items": {"A": {"url": "https://x.ca/a", "doi": None}}}
    # A was moved to a collection outside the tree since the last sync
    delta = [{"data": {"key": "A", "url": "https://x.ca/a", "collections": ["ELSEWHERE"]}}]
    client = FakeZoteroClient(tree=tree, items_by_version=[(delta, 11)])

    state = sync_zotero_seen_state(client, "ROOT1", seeded_state)

    assert "A" not in state["items"]


def test_post_to_groq_retries_on_transient_errors_then_succeeds():
    ok_response = _verdict_response({"ok": True})
    responses = [FakeGroqResponse(503), FakeGroqResponse(429), ok_response]
    with patch("sourcing.filter_candidates.requests.post", side_effect=responses), \
         patch("sourcing.filter_candidates.time.sleep"):
        resp = filter_candidates_module._post_to_groq({})
    assert resp is ok_response


def test_post_to_groq_does_not_retry_non_retriable_status():
    with patch("sourcing.filter_candidates.requests.post", return_value=FakeGroqResponse(401)), \
         patch("sourcing.filter_candidates.time.sleep") as mock_sleep:
        with pytest.raises(requests.exceptions.HTTPError):
            filter_candidates_module._post_to_groq({})
    mock_sleep.assert_not_called()


def test_post_to_groq_gives_up_after_max_retries():
    with patch("sourcing.filter_candidates.requests.post", return_value=FakeGroqResponse(503)), \
         patch("sourcing.filter_candidates.time.sleep"):
        with pytest.raises(requests.exceptions.HTTPError):
            filter_candidates_module._post_to_groq({})


def test_filter_candidates_leaves_candidate_unmarked_when_judge_fails():
    c = Candidate(title="facial recognition guidance", url="https://x.ca/a", source_family="government")
    state = {}
    with patch("sourcing.filter_candidates.judge_candidate", side_effect=RuntimeError("boom")):
        passed = filter_candidates([c], load_taxonomy(), state)
    assert passed == []
    assert dedup_key(c) not in state  # left for next run, not permanently rejected


def test_filter_candidates_circuit_breaker_stops_after_consecutive_failures(monkeypatch):
    monkeypatch.setattr(filter_candidates_module, "_MIN_SECONDS_BETWEEN_CALLS", 0)
    candidates = [
        Candidate(title=f"facial recognition item {i}", url=f"https://x.ca/{i}", source_family="government")
        for i in range(5)
    ]
    attempted = []

    def always_fails(c, taxonomy, existing_tags=None):
        attempted.append(c.url)
        raise RuntimeError("simulated outage")

    state = {}
    with patch("sourcing.filter_candidates.judge_candidate", side_effect=always_fails):
        passed = filter_candidates(candidates, load_taxonomy(), state)

    # breaker trips at _CONSECUTIVE_FAILURE_CIRCUIT_BREAKER (3) — stops
    # instead of grinding through all 5
    assert len(attempted) == filter_candidates_module._CONSECUTIVE_FAILURE_CIRCUIT_BREAKER
    assert passed == []
    assert len(state) == 0  # none of them get permanently marked


def test_filter_candidates_caps_judge_calls_and_leaves_rest_unmarked(monkeypatch):
    monkeypatch.setattr(filter_candidates_module, "MAX_JUDGE_CALLS_PER_RUN", 1)
    monkeypatch.setattr(filter_candidates_module, "_MIN_SECONDS_BETWEEN_CALLS", 0)
    candidates = [
        Candidate(title=f"facial recognition item {i}", url=f"https://x.ca/{i}", source_family="government")
        for i in range(3)
    ]
    judged_urls = []

    def fake_judge(c, taxonomy, existing_tags=None):
        judged_urls.append(c.url)
        return {"relevant": True, "reason": "", "practice_tags": [], "ethics_tags": [], "jurisdiction_tags": []}

    state = {}
    with patch("sourcing.filter_candidates.judge_candidate", side_effect=fake_judge):
        passed = filter_candidates(candidates, load_taxonomy(), state)

    assert len(judged_urls) == 1  # capped at MAX_JUDGE_CALLS_PER_RUN
    assert len(passed) == 1
    # the deferred candidates aren't marked in state, so they're retried next run
    assert len(state) == 0
