import importlib.util
import json
import tarfile
from pathlib import Path

import pytest


BUILD_DIR = Path(__file__).resolve().parents[1]
PUBLISH_FILE = BUILD_DIR / "publish_release.py"

spec = importlib.util.spec_from_file_location(
    "publish_release_module",
    PUBLISH_FILE,
)

publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def make_valid_build(tmp_path: Path) -> Path:
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    metadata = {
        "build_id": "20260922T203002Z",
        "built_at": "2026-09-22T20:30:02+00:00",
        "model": "bge-base-en-v1.5",
        "dim": 768,
        "approved_sources": 406,
        "successfully_indexed_sources": 308,
        "sources_without_final_chunks": 98,
        "recorded_source_failure_events": 98,
        "final_chunks": 5142,
        "skipped_chunk_events": 92,
    }

    files = {
        "faiss.index": b"fake-faiss-index",
        "chunks.json": b"[]",
        "source_summary.json": b"{}",
        "failed_sources.json": b"[]",
        "skipped_chunks.json": b"[]",
        "build_meta.json": json.dumps(metadata).encode("utf-8"),
    }

    for filename, contents in files.items():
        (build_dir / filename).write_bytes(contents)

    return build_dir


def test_validate_files_rejects_missing_artifact(tmp_path):
    build_dir = make_valid_build(tmp_path)

    (build_dir / "faiss.index").unlink()

    with pytest.raises(RuntimeError, match="faiss.index"):
        publisher.validate_files(build_dir)


def test_load_metadata_rejects_old_schema(tmp_path):
    build_dir = make_valid_build(tmp_path)

    old_metadata = {
        "model": "bge-base-en-v1.5",
        "dim": 768,
        "num_chunks": 5142,
        "num_sources": 332,
        "num_failed_sources": 98,
        "num_skipped_chunks": 92,
    }

    (build_dir / "build_meta.json").write_text(
        json.dumps(old_metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="missing required fields",
    ):
        publisher.load_metadata(build_dir)


def test_make_archive_contains_exact_required_files(tmp_path):
    build_dir = make_valid_build(tmp_path)
    archive_path = tmp_path / "prrism-index.tar.gz"

    publisher.make_archive(build_dir, archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        names = set(tar.getnames())

    assert names == publisher.REQUIRED_FILES


def test_release_identity_uses_build_id(tmp_path):
    build_dir = make_valid_build(tmp_path)
    metadata = publisher.load_metadata(build_dir)

    assert (
        publisher.release_tag(metadata["build_id"])
        == "index-20260922T203002Z"
    )

    assert (
        publisher.release_title(metadata["build_id"])
        == "PRRISM Index — 20260922T203002Z"
    )