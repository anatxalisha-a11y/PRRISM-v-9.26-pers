import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


BUILD_DIR = Path(__file__).resolve().parents[1]

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))


FETCH_FILE = BUILD_DIR / "fetch_release.py"

spec = importlib.util.spec_from_file_location(
    "fetch_release_module",
    FETCH_FILE,
)

fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def make_archive(path: Path, filenames):
    """Create a tiny tar.gz containing the requested files."""

    with tarfile.open(path, "w:gz") as archive:
        for filename in filenames:
            data = b"test"

            info = tarfile.TarInfo(filename)
            info.size = len(data)

            archive.addfile(
                info,
                io.BytesIO(data),
            )


def test_release_prefix_is_explicit():
    assert fetcher.RELEASE_PREFIX == "index-"
    assert fetcher.ASSET_PREFIX == "prrism-index-"
    assert fetcher.ASSET_SUFFIX == ".tar.gz"


def test_extract_archive_requires_exact_artifacts(tmp_path):
    archive_path = tmp_path / "build.tar.gz"

    make_archive(
        archive_path,
        fetcher.REQUIRED_FILES - {"chunks.json"},
    )

    with pytest.raises(
        fetcher.ReleaseFetchError,
        match="exactly the expected",
    ):
        fetcher.extract_archive(
            archive_path,
            tmp_path / "candidate",
        )


def test_extract_archive_accepts_expected_artifacts(tmp_path):
    archive_path = tmp_path / "build.tar.gz"
    destination = tmp_path / "candidate"

    make_archive(
        archive_path,
        fetcher.REQUIRED_FILES,
    )

    fetcher.extract_archive(
        archive_path,
        destination,
    )

    assert {
        path.name
        for path in destination.iterdir()
    } == fetcher.REQUIRED_FILES


def test_digest_verification_passes(tmp_path):
    archive_path = tmp_path / "artifact.tar.gz"
    archive_path.write_bytes(b"PRRISM")

    import hashlib

    digest = hashlib.sha256(b"PRRISM").hexdigest()

    fetcher.verify_digest(
        archive_path,
        f"sha256:{digest}",
    )


def test_digest_verification_rejects_mismatch(tmp_path):
    archive_path = tmp_path / "artifact.tar.gz"
    archive_path.write_bytes(b"PRRISM")

    with pytest.raises(
        fetcher.ReleaseFetchError,
        match="digest mismatch",
    ):
        fetcher.verify_digest(
            archive_path,
            "sha256:" + ("0" * 64),
        )


def test_find_latest_index_release(monkeypatch):
    releases = [
        {
            "tagName": "v1.0",
            "publishedAt": "2026-09-30T00:00:00Z",
            "isDraft": False,
            "isPrerelease": False,
        },
        {
            "tagName": "index-2026-09-20",
            "publishedAt": "2026-09-20T00:00:00Z",
            "isDraft": False,
            "isPrerelease": False,
        },
        {
            "tagName": "index-2026-09-22",
            "publishedAt": "2026-09-22T00:00:00Z",
            "isDraft": False,
            "isPrerelease": False,
        },
    ]

    monkeypatch.setattr(
        fetcher,
        "run_gh",
        lambda args: json.dumps(releases),
    )

    assert (
        fetcher.find_latest_index_release()
        == "index-2026-09-22"
    )


def test_prerelease_is_not_selected(monkeypatch):
    releases = [
        {
            "tagName": "index-future",
            "publishedAt": "2026-10-01T00:00:00Z",
            "isDraft": False,
            "isPrerelease": True,
        },
        {
            "tagName": "index-stable",
            "publishedAt": "2026-09-22T00:00:00Z",
            "isDraft": False,
            "isPrerelease": False,
        },
    ]

    monkeypatch.setattr(
        fetcher,
        "run_gh",
        lambda args: json.dumps(releases),
    )

    assert (
        fetcher.find_latest_index_release()
        == "index-stable"
    )