import importlib.util
import json
from pathlib import Path

import faiss
import numpy as np
import pytest


BUILD_DIR = Path(__file__).resolve().parents[1]
VALIDATOR_FILE = BUILD_DIR / "validate_build.py"

spec = importlib.util.spec_from_file_location(
    "validate_build_module",
    VALIDATOR_FILE,
)

validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

validate_build = validator.validate_build
BuildValidationError = validator.BuildValidationError


def create_valid_build(tmp_path):
    """Create a tiny but internally consistent fake build."""

    chunks = [
        {
            "source_id": "SRC1",
            "url": "https://example.com",
            "title": "Example Policy",
            "chunk_id": 0,
            "text": "Meaningful policy text.",
            "practice_tags": [],
            "ethics_tags": [],
            "jurisdiction_tags": [],
        }
    ]

    source_summary = {
        "SRC1": {
            "source_id": "SRC1",
            "url": "https://example.com",
            "title": "Example Policy",
        }
    }

    build_meta = {
        "model": "test-model",
        "dim": 4,
        "num_chunks": 1,
        "num_sources": 1,
        "num_failed_sources": 0,
        "num_skipped_chunks": 0,
    }

    index = faiss.IndexFlatIP(4)
    index.add(
        np.array(
            [[1.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
    )

    faiss.write_index(
        index,
        str(tmp_path / "faiss.index"),
    )

    files = {
        "chunks.json": chunks,
        "source_summary.json": source_summary,
        "failed_sources.json": [],
        "skipped_chunks.json": [],
        "build_meta.json": build_meta,
    }

    for filename, data in files.items():
        with (tmp_path / filename).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f)

    return tmp_path


def test_valid_build_passes(tmp_path):
    build_dir = create_valid_build(tmp_path)

    result = validate_build(build_dir)

    assert result["valid"] is True
    assert result["num_chunks"] == 1
    assert result["num_sources"] == 1
    assert result["dimension"] == 4


def test_missing_artifact_fails(tmp_path):
    build_dir = create_valid_build(tmp_path)

    (build_dir / "chunks.json").unlink()

    with pytest.raises(
        BuildValidationError,
        match="Missing build artifacts",
    ):
        validate_build(build_dir)


def test_faiss_chunk_mismatch_fails(tmp_path):
    build_dir = create_valid_build(tmp_path)

    with (build_dir / "chunks.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)

    chunks.append(dict(chunks[0], chunk_id=1))

    with (build_dir / "chunks.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(chunks, f)

    with pytest.raises(
        BuildValidationError,
        match="FAISS/chunk mismatch",
    ):
        validate_build(build_dir)


def test_dimension_mismatch_fails(tmp_path):
    build_dir = create_valid_build(tmp_path)

    meta_path = build_dir / "build_meta.json"

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["dim"] = 999

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(
        BuildValidationError,
        match="Embedding dimension mismatch",
    ):
        validate_build(build_dir)


def test_missing_chunk_metadata_fails(tmp_path):
    build_dir = create_valid_build(tmp_path)

    chunks_path = build_dir / "chunks.json"

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    del chunks[0]["jurisdiction_tags"]

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f)

    with pytest.raises(
        BuildValidationError,
        match="missing fields",
    ):
        validate_build(build_dir)


def test_unknown_source_id_fails(tmp_path):
    build_dir = create_valid_build(tmp_path)

    chunks_path = build_dir / "chunks.json"

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunks[0]["source_id"] = "DOES_NOT_EXIST"

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f)

    with pytest.raises(
        BuildValidationError,
        match="unknown source_id",
    ):
        validate_build(build_dir)


def test_empty_chunk_text_fails(tmp_path):
    build_dir = create_valid_build(tmp_path)

    chunks_path = build_dir / "chunks.json"

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunks[0]["text"] = "   "

    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f)

    with pytest.raises(
        BuildValidationError,
        match="empty text",
    ):
        validate_build(build_dir)