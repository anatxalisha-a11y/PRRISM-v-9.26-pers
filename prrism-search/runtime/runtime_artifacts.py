"""Cloud-safe PRRISM artifact bootstrap.

Downloads the newest validated PRRISM index release from the private
GitHub repository when no valid local build is available.
"""

import hashlib
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict

import requests


BASE_DIR = Path(__file__).resolve().parents[2]

OUT_DIR = BASE_DIR / "out"
CURRENT_DIR = OUT_DIR / "current"

REPO_OWNER = "anatxalisha-a11y"
REPO_NAME = "PRRISM-v-9.26"

RELEASE_PREFIX = "index-"
ASSET_PREFIX = "prrism-index-"
ASSET_SUFFIX = ".tar.gz"

GITHUB_API = "https://api.github.com"

REQUIRED_FILES = {
    "faiss.index",
    "chunks.json",
    "source_summary.json",
    "failed_sources.json",
    "skipped_chunks.json",
    "build_meta.json",
}


class RuntimeArtifactError(RuntimeError):
    """Raised when runtime artifacts cannot be safely prepared."""


def _github_token() -> str:
    """Return the read-only GitHub token used for release retrieval."""

    token = os.environ.get(
        "PRRISM_GITHUB_TOKEN",
        ""
    ).strip()

    if not token:
        raise RuntimeArtifactError(
            "PRRISM_GITHUB_TOKEN is not configured."
        )

    return token


def _headers(
    accept: str = "application/vnd.github+json",
) -> Dict[str, str]:
    """Build authenticated GitHub API headers."""

    return {
        "Accept": accept,
        "Authorization": (
            f"Bearer {_github_token()}"
        ),
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "PRRISM-runtime",
    }


def _request_json(
    url: str,
) -> Any:
    """GET JSON from GitHub with explicit error handling."""

    try:
        response = requests.get(
            url,
            headers=_headers(),
            timeout=30,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeArtifactError(
            f"GitHub API request failed: {url}"
        ) from exc

    try:
        return response.json()

    except ValueError as exc:
        raise RuntimeArtifactError(
            "GitHub returned invalid JSON."
        ) from exc


def find_latest_index_release() -> Dict[str, Any]:
    """Find the newest published PRRISM index release."""

    releases = _request_json(
        f"{GITHUB_API}/repos/"
        f"{REPO_OWNER}/{REPO_NAME}/releases"
        "?per_page=100"
    )

    candidates = [
        release
        for release in releases
        if (
            str(
                release.get("tag_name", "")
            ).startswith(RELEASE_PREFIX)
            and not release.get("draft", False)
            and not release.get(
                "prerelease",
                False,
            )
        )
    ]

    if not candidates:
        raise RuntimeArtifactError(
            "No published PRRISM index releases "
            "were found."
        )

    candidates.sort(
        key=lambda release: (
            release.get("published_at")
            or ""
        ),
        reverse=True,
    )

    return candidates[0]


def get_release_asset(
    release: Dict[str, Any],
) -> Dict[str, Any]:
    """Return the release's single PRRISM index asset."""

    assets = [
        asset
        for asset in release.get(
            "assets",
            [],
        )
        if (
            str(
                asset.get("name", "")
            ).startswith(ASSET_PREFIX)
            and str(
                asset.get("name", "")
            ).endswith(ASSET_SUFFIX)
        )
    ]

    if len(assets) != 1:
        raise RuntimeArtifactError(
            "Expected exactly one PRRISM index "
            f"asset, found {len(assets)}."
        )

    return assets[0]


def download_asset(
    asset: Dict[str, Any],
    destination: Path,
) -> Path:
    """Download a private GitHub release asset."""

    asset_id = asset.get("id")
    asset_name = asset.get("name")

    if not asset_id or not asset_name:
        raise RuntimeArtifactError(
            "Release asset metadata is incomplete."
        )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive_path = (
        destination / asset_name
    )

    url = (
        f"{GITHUB_API}/repos/"
        f"{REPO_OWNER}/{REPO_NAME}/"
        f"releases/assets/{asset_id}"
    )

    try:
        with requests.get(
            url,
            headers=_headers(
                "application/octet-stream"
            ),
            timeout=120,
            stream=True,
        ) as response:

            response.raise_for_status()

            with archive_path.open(
                "wb"
            ) as output:

                for block in (
                    response.iter_content(
                        chunk_size=1024 * 1024
                    )
                ):
                    if block:
                        output.write(block)

    except requests.RequestException as exc:
        raise RuntimeArtifactError(
            "Could not download the PRRISM "
            "release asset."
        ) from exc

    if not archive_path.is_file():
        raise RuntimeArtifactError(
            "Downloaded release asset was "
            "not created."
        )

    return archive_path


def verify_digest(
    archive_path: Path,
    expected_digest: str | None,
) -> None:
    """Verify GitHub's SHA-256 digest when available."""

    if not expected_digest:
        return

    if not expected_digest.startswith(
        "sha256:"
    ):
        raise RuntimeArtifactError(
            "Unsupported GitHub release digest."
        )

    expected = expected_digest.split(
        ":",
        1,
    )[1].lower()

    sha256 = hashlib.sha256()

    with archive_path.open("rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            sha256.update(block)

    actual = sha256.hexdigest()

    if actual != expected:
        raise RuntimeArtifactError(
            "Release asset SHA-256 digest "
            "does not match GitHub metadata."
        )


def extract_archive(
    archive_path: Path,
    destination: Path,
) -> None:
    """Safely extract exactly the expected build files."""

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        archive_path,
        "r:gz",
    ) as archive:

        members = archive.getmembers()

        member_names = {
            member.name.removeprefix("./")
            for member in members
            if member.isfile()
        }

        if member_names != REQUIRED_FILES:
            raise RuntimeArtifactError(
                "Release archive does not contain "
                "exactly the expected PRRISM "
                "build artifacts."
            )

        for member in members:

            normalized = Path(
                member.name.removeprefix("./")
            )

            if (
                normalized.is_absolute()
                or ".." in normalized.parts
            ):
                raise RuntimeArtifactError(
                    "Unsafe archive path: "
                    f"{member.name}"
                )

            if not member.isfile():
                continue

            target = (
                destination / normalized
            )

            source = archive.extractfile(
                member
            )

            if source is None:
                raise RuntimeArtifactError(
                    "Could not extract "
                    f"{member.name}."
                )

            with source:
                with target.open(
                    "wb"
                ) as output:

                    while True:

                        block = source.read(
                            1024 * 1024
                        )

                        if not block:
                            break

                        output.write(block)


def local_artifacts_present() -> bool:
    """Check whether all expected runtime artifacts exist."""

    return all(
        (
            CURRENT_DIR / filename
        ).is_file()
        for filename in REQUIRED_FILES
    )


def ensure_runtime_artifacts() -> Dict[str, Any]:
    """
    Ensure PRRISM has a local runtime index.

    Existing artifacts are left untouched. If they are absent, the
    newest published index release is downloaded, verified, validated,
    and activated.
    """

    if local_artifacts_present():

        return {
            "status": "existing",
            "path": str(CURRENT_DIR),
        }

    # Import these only when a bootstrap is actually necessary.
    # prrism-build is not a normal Python package, so expose it
    # explicitly for the existing build tooling.
    import sys

    build_tools = (
        BASE_DIR / "prrism-build"
    )

    if str(build_tools) not in sys.path:
        sys.path.insert(
            0,
            str(build_tools),
        )

    from publish_build import publish_build
    from validate_build import validate_build

    release = (
        find_latest_index_release()
    )

    tag = release.get(
        "tag_name"
    )

    asset = get_release_asset(
        release
    )

    with tempfile.TemporaryDirectory() as temp:

        temp_dir = Path(temp)

        archive_path = download_asset(
            asset,
            temp_dir / "download",
        )

        verify_digest(
            archive_path,
            asset.get("digest"),
        )

        candidate_dir = (
            temp_dir / "candidate"
        )

        extract_archive(
            archive_path,
            candidate_dir,
        )

        validation = validate_build(
            candidate_dir
        )

        result = publish_build(
            candidate_dir,
            OUT_DIR,
        )

    return {
        "status": "downloaded",
        "tag": tag,
        "num_sources": validation[
            "num_sources"
        ],
        "num_chunks": validation[
            "num_chunks"
        ],
        "dimension": validation[
            "dimension"
        ],
        **result,
    }