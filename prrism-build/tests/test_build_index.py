from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest


# ---------------------------------------------------------
# Import build_index.py from the hyphenated prrism-build dir
# ---------------------------------------------------------
BUILD_DIR = Path(__file__).resolve().parents[1]
BUILD_FILE = BUILD_DIR / "build_index.py"

# prrism-build is hyphenated, so make its modules importable explicitly.
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

spec = importlib.util.spec_from_file_location(
    "build_index_module",
    BUILD_FILE,
)

builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


Source = builder.Source


# ---------------------------------------------------------
# Test data
# ---------------------------------------------------------
@pytest.fixture
def source():
    return Source(
        source_id="TEST1234",
        url="https://example.com/policy",
        title="Example AI Policy",
        item_type="webpage",
        collections=("Policy",),
    )


@pytest.fixture
def tags():
    return {
        "practice_tags": {
            "human oversight": (
                "Human review and oversight of AI systems."
            ),
            "explainability": (
                "Methods for explaining AI system behaviour."
            ),
        },
        "ethics_tags": {
            "fairness": (
                "Fair and equitable treatment in AI systems."
            ),
            "accountability": (
                "Responsibility and accountability for AI systems."
            ),
        },
        "jurisdiction_tags": {
            "Canada": (
                "Canadian laws, regulations, policy, and governance."
            ),
            "European Union": (
                "European Union laws, regulations, policy, "
                "and governance."
            ),
        },
    }


class FakeZoteroClient:
    def download_attachment(self, item_key: str) -> bytes:
        raise AssertionError(
            "download_attachment should not be called for URL sources"
        )


@pytest.fixture
def zotero():
    return FakeZoteroClient()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def fake_process_url(url):
    text = (
        "This Canadian artificial intelligence policy discusses human "
        "oversight, accountability, fairness, explainability, and governance. "
    ) * 20

    return text, None


def fake_embed_resilient(texts, batch_size=32, label="texts"):
    """
    Return deterministic fake 4-dimensional embeddings.

    This lets us test the builder without Vector / vec-inf.
    """
    embeddings = []

    for i, _ in enumerate(texts):
        vec = np.array(
            [
                1.0 + i,
                2.0 + i,
                3.0 + i,
                4.0 + i,
            ],
            dtype=np.float32,
        )

        embeddings.append(vec.tolist())

    kept_indices = list(range(len(texts)))
    failures = []

    return embeddings, kept_indices, failures


# ---------------------------------------------------------
# Build-index tests
# ---------------------------------------------------------
def test_build_index_preserves_zotero_metadata(
    monkeypatch,
    source,
    tags,
    zotero,
):
    monkeypatch.setattr(
        builder,
        "process_url",
        fake_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    result = builder.build_index(
        [source],
        tags,
        zotero,
    )

    assert len(result["chunks"]) > 0

    first_chunk = result["chunks"][0]

    assert first_chunk["source_id"] == "TEST1234"
    assert first_chunk["url"] == "https://example.com/policy"
    assert first_chunk["title"] == "Example AI Policy"
    assert first_chunk["collections"] == ["Policy"]


def test_chunks_contain_all_three_taxonomies(
    monkeypatch,
    source,
    tags,
    zotero,
):
    monkeypatch.setattr(
        builder,
        "process_url",
        fake_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    result = builder.build_index(
        [source],
        tags,
        zotero,
    )

    for chunk in result["chunks"]:
        assert "practice_tags" in chunk
        assert "ethics_tags" in chunk
        assert "jurisdiction_tags" in chunk

        assert isinstance(chunk["practice_tags"], list)
        assert isinstance(chunk["ethics_tags"], list)
        assert isinstance(chunk["jurisdiction_tags"], list)


def test_source_summary_preserves_zotero_metadata(
    monkeypatch,
    source,
    tags,
    zotero,
):
    monkeypatch.setattr(
        builder,
        "process_url",
        fake_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    result = builder.build_index(
        [source],
        tags,
        zotero,
    )

    summary = result["source_summary"]["TEST1234"]

    assert summary["source_id"] == "TEST1234"
    assert summary["url"] == "https://example.com/policy"
    assert summary["title"] == "Example AI Policy"
    assert summary["item_type"] == "webpage"
    assert summary["collections"] == ["Policy"]


def test_source_summary_contains_all_taxonomies(
    monkeypatch,
    source,
    tags,
    zotero,
):
    monkeypatch.setattr(
        builder,
        "process_url",
        fake_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    result = builder.build_index(
        [source],
        tags,
        zotero,
    )

    summary = result["source_summary"]["TEST1234"]

    assert "practice_tags" in summary
    assert "ethics_tags" in summary
    assert "jurisdiction_tags" in summary


def test_faiss_count_matches_chunk_count(
    monkeypatch,
    source,
    tags,
    zotero,
):
    monkeypatch.setattr(
        builder,
        "process_url",
        fake_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    result = builder.build_index(
        [source],
        tags,
        zotero,
    )

    assert result["faiss_index"].ntotal == len(result["chunks"])


def test_embedding_dimension_matches_faiss(
    monkeypatch,
    source,
    tags,
    zotero,
):
    monkeypatch.setattr(
        builder,
        "process_url",
        fake_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    result = builder.build_index(
        [source],
        tags,
        zotero,
    )

    assert result["dim"] == 4
    assert result["faiss_index"].d == 4


def test_failed_source_does_not_enter_index(
    monkeypatch,
    source,
    tags,
    zotero,
):
    def failing_process_url(url):
        return None, "Fake extraction failure"

    monkeypatch.setattr(
        builder,
        "process_url",
        failing_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    with pytest.raises(
        ValueError,
        match="No embeddable chunks were created",
    ):
        builder.build_index(
            [source],
            tags,
            zotero,
        )


def test_one_failed_source_does_not_break_successful_source(
    monkeypatch,
    source,
    tags,
    zotero,
):
    second_source = Source(
        source_id="FAIL5678",
        url="https://example.com/failing",
        title="Failing Source",
        item_type="webpage",
        collections=("Benchmarks",),
    )

    def mixed_process_url(url):
        if url == second_source.url:
            return None, "Fake extraction failure"

        return fake_process_url(url)

    monkeypatch.setattr(
        builder,
        "process_url",
        mixed_process_url,
    )
    monkeypatch.setattr(
        builder,
        "embed_resilient",
        fake_embed_resilient,
    )

    result = builder.build_index(
        [source, second_source],
        tags,
        zotero,
    )

    assert "TEST1234" in result["source_summary"]
    assert "FAIL5678" not in result["source_summary"]

    assert len(result["failed_sources"]) == 1
    assert result["failed_sources"][0]["url"] == second_source.url

    assert all(
        chunk["source_id"] == "TEST1234"
        for chunk in result["chunks"]
    )


# ---------------------------------------------------------
# Stored Zotero attachment tests
# ---------------------------------------------------------
def test_process_source_downloads_stored_pdf_from_zotero(
    monkeypatch,
):
    stored_source = Source(
        source_id="PDF12345",
        url="",
        title="Stored Policy PDF",
        item_type="attachment",
        collections=("Policy",),
        content_type="application/pdf",
        link_mode="imported_file",
        attachment_key="PDF12345",
    )

    class StoredPdfZoteroClient:
        def __init__(self):
            self.requested_key = None

        def download_attachment(self, item_key):
            self.requested_key = item_key
            return b"%PDF-fake-pdf-data"

    fake_zotero = StoredPdfZoteroClient()

    monkeypatch.setattr(
        builder,
        "pdf_to_text",
        lambda data: "Extracted text from stored Zotero PDF.",
    )

    text, error = builder.process_source(
        stored_source,
        fake_zotero,
    )

    assert error is None
    assert text == "Extracted text from stored Zotero PDF."
    assert fake_zotero.requested_key == "PDF12345"


def test_process_source_detects_stored_pdf_by_magic_bytes(
    monkeypatch,
):
    stored_source = Source(
        source_id="PDFMAGIC",
        url="",
        title="Stored PDF With Generic Content Type",
        item_type="attachment",
        collections=("Policy",),
        content_type="application/octet-stream",
        link_mode="imported_file",
        attachment_key="PDFMAGIC",
    )

    class StoredPdfZoteroClient:
        def download_attachment(self, item_key):
            assert item_key == "PDFMAGIC"
            return b"%PDF-fake-pdf-data"

    monkeypatch.setattr(
        builder,
        "pdf_to_text",
        lambda data: "PDF detected from magic bytes.",
    )

    text, error = builder.process_source(
        stored_source,
        StoredPdfZoteroClient(),
    )

    assert error is None
    assert text == "PDF detected from magic bytes."


def test_process_source_rejects_non_pdf_stored_attachment():
    stored_source = Source(
        source_id="TXT12345",
        url="",
        title="Stored Text Attachment",
        item_type="attachment",
        collections=("Policy",),
        content_type="text/plain",
        link_mode="imported_file",
        attachment_key="TXT12345",
    )

    class StoredTextZoteroClient:
        def download_attachment(self, item_key):
            assert item_key == "TXT12345"
            return b"This is not a PDF."

    text, error = builder.process_source(
        stored_source,
        StoredTextZoteroClient(),
    )

    assert text is None
    assert error is not None
    assert "Unsupported Zotero stored attachment" in error


def test_process_source_url_source_uses_process_url(
    monkeypatch,
    source,
    zotero,
):
    calls = []

    def tracking_process_url(url):
        calls.append(url)
        return "Extracted URL text", None

    monkeypatch.setattr(
        builder,
        "process_url",
        tracking_process_url,
    )

    text, error = builder.process_source(
        source,
        zotero,
    )

    assert error is None
    assert text == "Extracted URL text"
    assert calls == ["https://example.com/policy"]


# ---------------------------------------------------------
# Candidate build directory tests
# ---------------------------------------------------------
def test_candidate_directories_are_unique(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        builder,
        "BUILDS_DIR",
        tmp_path,
    )

    first = builder.create_candidate_dir()

    assert first.exists()
    assert first.parent == tmp_path


# ---------------------------------------------------------
# Taxonomy tests
# ---------------------------------------------------------
def test_tag_value_to_text_supports_rich_jurisdiction():
    value = {
        "definition": (
            "Policies relating to facial recognition in Canada."
        ),
        "aliases": [
            "Canada",
            "Canadian",
        ],
        "jurisdictions": [
            "Ontario",
            "Quebec",
        ],
    }

    result = builder.tag_value_to_text(
        "Canada",
        value,
    )

    assert "Canada" in result
    assert (
        "Policies relating to facial recognition in Canada."
        in result
    )
    assert "Canadian" in result
    assert "Ontario" in result
    assert "Quebec" in result


def test_tag_value_to_text_supports_string_definition():
    result = builder.tag_value_to_text(
        "fairness",
        "Ensuring equitable treatment and outcomes.",
    )

    assert result == "Ensuring equitable treatment and outcomes."


# ---------------------------------------------------------
# URL extraction tests
# ---------------------------------------------------------
def test_process_url_uses_pdf_extractor_for_pdf_content(
    monkeypatch,
):
    class FakeResponse:
        headers = {
            "Content-Type": "application/pdf",
        }
        content = b"%PDF-fake"
        text = "%PDF-fake"
        url = "https://example.com/document.pdf"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        builder.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    monkeypatch.setattr(
        builder,
        "pdf_to_text",
        lambda data: "Extracted PDF text",
    )

    text, error = builder.process_url(
        "https://example.com/document.pdf"
    )

    assert error is None
    assert text == "Extracted PDF text"


def test_process_url_detects_pdf_by_magic_bytes(
    monkeypatch,
):
    class FakeResponse:
        headers = {
            "Content-Type": "application/octet-stream",
        }
        content = b"%PDF-fake"
        text = "%PDF-fake"
        url = "https://example.com/download?id=123"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        builder.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    monkeypatch.setattr(
        builder,
        "pdf_to_text",
        lambda data: "PDF detected by signature",
    )

    text, error = builder.process_url(
        "https://example.com/download?id=123"
    )

    assert error is None
    assert text == "PDF detected by signature"


def test_process_url_uses_html_extractor_for_html(
    monkeypatch,
):
    class FakeResponse:
        headers = {
            "Content-Type": "text/html",
        }
        content = b"<html><body>Hello PRRISM</body></html>"
        text = "<html><body>Hello PRRISM</body></html>"
        url = "https://example.com/page"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        builder.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    monkeypatch.setattr(
        builder,
        "html_to_text",
        lambda html, url="": "Extracted HTML text",
    )

    text, error = builder.process_url(
        "https://example.com/page"
    )

    assert error is None
    assert text == "Extracted HTML text"


# ---------------------------------------------------------
# Corpus quality-gate tests
# ---------------------------------------------------------
def test_corpus_quality_gate_passes_at_threshold():
    result = builder.validate_corpus_quality(
        total_sources=4,
        successful_sources=3,
        min_success_rate=0.75,
    )

    assert result["total_sources"] == 4
    assert result["successful_sources"] == 3
    assert result["failed_sources"] == 1
    assert result["success_rate"] == 0.75


def test_corpus_quality_gate_fails_below_threshold():
    with pytest.raises(
        ValueError,
        match="Corpus quality gate failed",
    ):
        builder.validate_corpus_quality(
            total_sources=4,
            successful_sources=2,
            min_success_rate=0.75,
        )


def test_corpus_quality_gate_rejects_empty_corpus():
    with pytest.raises(
        ValueError,
        match="zero approved sources",
    ):
        builder.validate_corpus_quality(
            total_sources=0,
            successful_sources=0,
        )