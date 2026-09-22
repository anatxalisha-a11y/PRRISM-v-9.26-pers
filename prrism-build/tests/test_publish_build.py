import importlib.util
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest


BUILD_DIR = Path(__file__).resolve().parents[1]

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))


PUBLISH_FILE = BUILD_DIR / "publish_build.py"

spec = importlib.util.spec_from_file_location(
    "publish_build_module",
    PUBLISH_FILE,
)

publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def create_build(directory: Path, title: str):
    directory.mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "source_id": "SRC1",
            "url": "https://example.com",
            "title": title,
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
            "title": title,
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
        str(directory / "faiss.index"),
    )

    files = {
        "chunks.json": chunks,
        "source_summary.json": source_summary,
        "failed_sources.json": [],
        "skipped_chunks.json": [],
        "build_meta.json": build_meta,
    }

    for filename, data in files.items():
        with (directory / filename).open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f)

    return directory


def read_title(build_dir: Path) -> str:
    with (build_dir / "chunks.json").open(
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)

    return chunks[0]["title"]


def test_first_build_can_be_published(tmp_path):
    candidate = create_build(
        tmp_path / "candidate",
        "Version 1",
    )

    out_dir = tmp_path / "out"

    result = publisher.publish_build(
        candidate,
        out_dir,
    )

    assert result["valid"] is True
    assert (out_dir / "current").is_dir()
    assert read_title(out_dir / "current") == "Version 1"


def test_new_build_replaces_current_build(tmp_path):
    out_dir = tmp_path / "out"

    first = create_build(
        tmp_path / "candidate1",
        "Version 1",
    )

    second = create_build(
        tmp_path / "candidate2",
        "Version 2",
    )

    publisher.publish_build(first, out_dir)
    publisher.publish_build(second, out_dir)

    assert read_title(out_dir / "current") == "Version 2"


def test_invalid_candidate_does_not_replace_current(tmp_path):
    out_dir = tmp_path / "out"

    good = create_build(
        tmp_path / "good",
        "Working Version",
    )

    publisher.publish_build(good, out_dir)

    bad = create_build(
        tmp_path / "bad",
        "Broken Version",
    )

    # Deliberately corrupt candidate.
    (bad / "chunks.json").unlink()

    with pytest.raises(Exception):
        publisher.publish_build(
            bad,
            out_dir,
        )

    # Critical invariant:
    # old working build must still be active.
    assert read_title(out_dir / "current") == "Working Version"


def test_candidate_is_not_modified(tmp_path):
    candidate = create_build(
        tmp_path / "candidate",
        "Version 1",
    )

    out_dir = tmp_path / "out"

    publisher.publish_build(
        candidate,
        out_dir,
    )

    assert candidate.exists()
    assert read_title(candidate) == "Version 1"


def test_no_staging_directories_remain(tmp_path):
    candidate = create_build(
        tmp_path / "candidate",
        "Version 1",
    )

    out_dir = tmp_path / "out"

    publisher.publish_build(
        candidate,
        out_dir,
    )

    leftovers = [
        path.name
        for path in out_dir.iterdir()
        if path.name.startswith(".staging-")
        or path.name.startswith(".previous-")
    ]

    assert leftovers == []