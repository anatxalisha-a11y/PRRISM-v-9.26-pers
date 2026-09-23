import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from validate_build import validate_build


REQUIRED_FILES = {
    "faiss.index",
    "chunks.json",
    "source_summary.json",
    "failed_sources.json",
    "skipped_chunks.json",
    "build_meta.json",
}

REQUIRED_META_FIELDS = {
    "build_id",
    "built_at",
    "model",
    "dim",
    "approved_sources",
    "successfully_indexed_sources",
    "sources_without_final_chunks",
    "recorded_source_failure_events",
    "final_chunks",
    "skipped_chunk_events",
}


def load_metadata(build_dir: Path) -> dict:
    meta_path = build_dir / "build_meta.json"

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    missing = REQUIRED_META_FIELDS - metadata.keys()

    if missing:
        raise RuntimeError(
            "build_meta.json is missing required fields: "
            + ", ".join(sorted(missing))
        )

    return metadata


def validate_files(build_dir: Path) -> None:
    missing = [
        filename
        for filename in REQUIRED_FILES
        if not (build_dir / filename).is_file()
    ]

    if missing:
        raise RuntimeError(
            "Build is missing required files: "
            + ", ".join(sorted(missing))
        )


def make_archive(build_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for filename in sorted(REQUIRED_FILES):
            tar.add(
                build_dir / filename,
                arcname=filename,
            )


def release_tag(build_id: str) -> str:
    return f"index-{build_id}"


def release_title(build_id: str) -> str:
    return f"PRRISM Index — {build_id}"


def release_notes(metadata: dict) -> str:
    return "\n".join(
        [
            "Validated PRRISM retrieval index.",
            "",
            f"Build ID: {metadata['build_id']}",
            f"Approved sources: {metadata['approved_sources']}",
            (
                "Successfully indexed sources: "
                f"{metadata['successfully_indexed_sources']}"
            ),
            f"Final embedded chunks: {metadata['final_chunks']}",
            f"Embedding model: {metadata['model']}",
            f"Embedding dimension: {metadata['dim']}",
            "Built and validated on the Vector Killarney cluster.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "build_dir",
        type=Path,
        help="Path to the validated PRRISM build directory.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and package without creating a GitHub Release.",
    )

    args = parser.parse_args()

    build_dir = args.build_dir.resolve()

    validate_files(build_dir)

    print("Running build integrity validation...")
    validation = validate_build(build_dir)

    print(
        f"Build validation passed: "
        f"{validation['num_sources']} sources, "
        f"{validation['num_chunks']} chunks, "
        f"dimension {validation['dimension']}."
    )

    metadata = load_metadata(build_dir)

    build_id = metadata["build_id"]
    tag = release_tag(build_id)

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = (
            Path(temp_dir) / f"prrism-index-{build_id}.tar.gz"
        )

        make_archive(build_dir, archive_path)

        print(f"Build ID: {build_id}")
        print(f"Release tag: {tag}")
        print(f"Archive: {archive_path}")
        print(
            f"Successfully indexed sources: "
            f"{metadata['successfully_indexed_sources']}"
        )
        print(f"Final chunks: {metadata['final_chunks']}")

        if args.dry_run:
            print("\nDry run complete. No GitHub Release created.")
            return

        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                str(archive_path),
                "--title",
                release_title(build_id),
                "--notes",
                release_notes(metadata),
            ],
            check=True,
        )

        print(f"\nPublished GitHub Release: {tag}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Release publication failed: {exc}", file=sys.stderr)
        sys.exit(1)