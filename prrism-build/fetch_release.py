import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from publish_build import publish_build
from validate_build import validate_build


REPO = "anatxalisha-a11y/PRRISM-v-9.26"
RELEASE_PREFIX = "index-"
ASSET_PREFIX = "prrism-index-"
ASSET_SUFFIX = ".tar.gz"

REQUIRED_FILES = {
    "faiss.index",
    "chunks.json",
    "source_summary.json",
    "failed_sources.json",
    "skipped_chunks.json",
    "build_meta.json",
}


class ReleaseFetchError(RuntimeError):
    """Raised when a PRRISM release cannot be safely retrieved."""


def run_gh(args: List[str]) -> str:
    """Run a GitHub CLI command and return stdout."""

    try:
        result = subprocess.run(
            ["gh", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise ReleaseFetchError(
            "GitHub CLI ('gh') is not installed."
        ) from e
    except subprocess.CalledProcessError as e:
        message = e.stderr.strip() or e.stdout.strip()
        raise ReleaseFetchError(
            f"GitHub CLI command failed: {message}"
        ) from e

    return result.stdout


def find_latest_index_release() -> str:
    """
    Find the newest published, non-draft, non-prerelease
    PRRISM index release.
    """

    output = run_gh([
        "release",
        "list",
        "--repo",
        REPO,
        "--limit",
        "100",
        "--json",
        "tagName,publishedAt,isDraft,isPrerelease",
    ])

    releases = json.loads(output)

    candidates = [
        release
        for release in releases
        if release["tagName"].startswith(RELEASE_PREFIX)
        and not release["isDraft"]
        and not release["isPrerelease"]
    ]

    if not candidates:
        raise ReleaseFetchError(
            "No published PRRISM index releases were found."
        )

    candidates.sort(
        key=lambda release: release["publishedAt"],
        reverse=True,
    )

    return candidates[0]["tagName"]


def get_release_asset(tag: str) -> Dict[str, Any]:
    """Return metadata for the release's single PRRISM index asset."""

    output = run_gh([
        "release",
        "view",
        tag,
        "--repo",
        REPO,
        "--json",
        "assets",
    ])

    release = json.loads(output)

    assets = [
        asset
        for asset in release.get("assets", [])
        if asset["name"].startswith(ASSET_PREFIX)
        and asset["name"].endswith(ASSET_SUFFIX)
    ]

    if len(assets) != 1:
        raise ReleaseFetchError(
            f"Expected exactly one PRRISM index asset for {tag}, "
            f"found {len(assets)}."
        )

    return assets[0]


def download_asset(
    tag: str,
    asset_name: str,
    destination: Path,
) -> Path:
    """Download one release asset using authenticated GitHub CLI."""

    destination.mkdir(parents=True, exist_ok=True)

    run_gh([
        "release",
        "download",
        tag,
        "--repo",
        REPO,
        "--pattern",
        asset_name,
        "--dir",
        str(destination),
    ])

    archive_path = destination / asset_name

    if not archive_path.is_file():
        raise ReleaseFetchError(
            f"Downloaded asset was not found: {archive_path}"
        )

    return archive_path


def verify_digest(
    archive_path: Path,
    expected_digest: str | None,
) -> None:
    """Verify GitHub's SHA-256 digest when one is available."""

    if not expected_digest:
        return

    if not expected_digest.startswith("sha256:"):
        raise ReleaseFetchError(
            f"Unsupported release digest: {expected_digest}"
        )

    expected = expected_digest.split(":", 1)[1].lower()

    sha256 = hashlib.sha256()

    with archive_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    actual = sha256.hexdigest()

    if actual != expected:
        raise ReleaseFetchError(
            "Release asset SHA-256 digest mismatch."
        )


def extract_archive(
    archive_path: Path,
    destination: Path,
) -> None:
    """
    Safely extract a PRRISM build archive.

    Reject paths outside the extraction directory and require
    exactly the expected artifact files at the archive root.
    """

    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()

        member_names = {
            member.name.removeprefix("./")
            for member in members
            if member.isfile()
        }

        if member_names != REQUIRED_FILES:
            raise ReleaseFetchError(
                "Release archive does not contain exactly the "
                "expected PRRISM build artifacts."
            )

        for member in members:
            normalized = Path(
                member.name.removeprefix("./")
            )

            if (
                normalized.is_absolute()
                or ".." in normalized.parts
            ):
                raise ReleaseFetchError(
                    f"Unsafe archive path: {member.name}"
                )

            if not member.isfile():
                continue

            target = destination / normalized

            with archive.extractfile(member) as source:
                if source is None:
                    raise ReleaseFetchError(
                        f"Could not extract {member.name}."
                    )

                with target.open("wb") as output:
                    while True:
                        block = source.read(1024 * 1024)

                        if not block:
                            break

                        output.write(block)


def fetch_release(
    out_dir: Path,
    tag: str | None = None,
) -> Dict[str, Any]:
    """Download, verify, validate, and activate a PRRISM release."""

    if tag is None:
        tag = find_latest_index_release()

    asset = get_release_asset(tag)

    asset_name = asset["name"]
    expected_digest = asset.get("digest")

    print(f"Selected release: {tag}")
    print(f"Asset: {asset_name}")

    with tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)

        archive_path = download_asset(
            tag,
            asset_name,
            temp_dir / "download",
        )

        print("Verifying release digest...")
        verify_digest(
            archive_path,
            expected_digest,
        )

        candidate_dir = temp_dir / "candidate"

        print("Extracting release...")
        extract_archive(
            archive_path,
            candidate_dir,
        )

        print("Validating downloaded build...")
        validation = validate_build(candidate_dir)

        print(
            f"Downloaded build validated: "
            f"{validation['num_sources']} sources, "
            f"{validation['num_chunks']} chunks, "
            f"dimension {validation['dimension']}."
        )

        print("Activating build...")
        result = publish_build(
            candidate_dir,
            Path(out_dir),
        )

    print("PRRISM index is ready.")

    return {
        "tag": tag,
        **result,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve and activate a validated PRRISM "
            "index release."
        )
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out",
        help="PRRISM output directory.",
    )

    parser.add_argument(
        "--tag",
        help=(
            "Specific release tag to retrieve. "
            "Defaults to the newest published index-* release."
        ),
    )

    args = parser.parse_args()

    result = fetch_release(
        args.out_dir,
        tag=args.tag,
    )

    print()
    print("Active build:")
    print(f"  Release:   {result['tag']}")
    print(f"  Sources:   {result['num_sources']}")
    print(f"  Chunks:    {result['num_chunks']}")
    print(f"  Dimension: {result['dimension']}")
    print(f"  Model:     {result['model']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(
            f"PRRISM release retrieval failed: {e}",
            file=sys.stderr,
        )
        sys.exit(1)