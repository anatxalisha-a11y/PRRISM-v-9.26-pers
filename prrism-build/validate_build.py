import json
from pathlib import Path
from typing import Dict, Any

import faiss


REQUIRED_CHUNK_FIELDS = {
    "source_id",
    "url",
    "title",
    "chunk_id",
    "text",
    "practice_tags",
    "ethics_tags",
    "jurisdiction_tags",
}


class BuildValidationError(RuntimeError):
    """Raised when a candidate PRRISM build is invalid."""


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise BuildValidationError(
            f"Could not read {path.name}: {e}"
        ) from e


def validate_build(build_dir: Path) -> Dict[str, Any]:
    """
    Validate a completed PRRISM build directory.

    Returns a validation summary if valid.
    Raises BuildValidationError if invalid.
    """

    build_dir = Path(build_dir)

    required_files = {
        "faiss.index",
        "chunks.json",
        "source_summary.json",
        "failed_sources.json",
        "skipped_chunks.json",
        "build_meta.json",
    }

    # 1. Required artifacts must exist
    missing_files = sorted(
        name
        for name in required_files
        if not (build_dir / name).is_file()
    )

    if missing_files:
        raise BuildValidationError(
            f"Missing build artifacts: {missing_files}"
        )

    # 2. Artifacts must actually be readable
    try:
        index = faiss.read_index(
            str(build_dir / "faiss.index")
        )
    except Exception as e:
        raise BuildValidationError(
            f"Could not load faiss.index: {e}"
        ) from e

    chunks = _load_json(build_dir / "chunks.json")
    source_summary = _load_json(
        build_dir / "source_summary.json"
    )
    build_meta = _load_json(build_dir / "build_meta.json")

    # 3. Basic structure
    if not isinstance(chunks, list):
        raise BuildValidationError(
            "chunks.json must contain a list."
        )

    if not chunks:
        raise BuildValidationError(
            "Build contains zero chunks."
        )

    if not isinstance(source_summary, dict):
        raise BuildValidationError(
            "source_summary.json must contain an object."
        )

    if not source_summary:
        raise BuildValidationError(
            "Build contains zero successful sources."
        )

    if not isinstance(build_meta, dict):
        raise BuildValidationError(
            "build_meta.json must contain an object."
        )

    # 4. Critical FAISS <-> chunks invariant
    if index.ntotal != len(chunks):
        raise BuildValidationError(
            "FAISS/chunk mismatch: "
            f"FAISS contains {index.ntotal} vectors but "
            f"chunks.json contains {len(chunks)} chunks."
        )

    # 5. Dimensions must be sane
    if index.d <= 0:
        raise BuildValidationError(
            f"Invalid FAISS dimension: {index.d}"
        )

    meta_dim = build_meta.get("dim")

    if meta_dim != index.d:
        raise BuildValidationError(
            "Embedding dimension mismatch: "
            f"build_meta.json says {meta_dim}, "
            f"FAISS says {index.d}."
        )

    # 6. build_meta counts must describe actual artifacts
    meta_chunks = build_meta.get("num_chunks")

    if meta_chunks != len(chunks):
        raise BuildValidationError(
            "Chunk count mismatch: "
            f"build_meta.json says {meta_chunks}, "
            f"chunks.json contains {len(chunks)}."
        )

    meta_sources = build_meta.get("num_sources")

    if meta_sources != len(source_summary):
        raise BuildValidationError(
            "Source count mismatch: "
            f"build_meta.json says {meta_sources}, "
            f"source_summary.json contains "
            f"{len(source_summary)}."
        )

    # 7. Every chunk must have the runtime metadata we expect
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise BuildValidationError(
                f"Chunk {i} is not an object."
            )

        missing_fields = REQUIRED_CHUNK_FIELDS - set(chunk)

        if missing_fields:
            raise BuildValidationError(
                f"Chunk {i} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not chunk["text"].strip():
            raise BuildValidationError(
                f"Chunk {i} has empty text."
            )

        source_id = chunk["source_id"]

        if source_id not in source_summary:
            raise BuildValidationError(
                f"Chunk {i} references unknown "
                f"source_id '{source_id}'."
            )

    return {
        "valid": True,
        "num_chunks": len(chunks),
        "num_sources": len(source_summary),
        "dimension": index.d,
        "model": build_meta.get("model"),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate PRRISM build artifacts."
    )

    parser.add_argument(
        "build_dir",
        type=Path,
        help="Directory containing candidate build artifacts.",
    )

    args = parser.parse_args()

    result = validate_build(args.build_dir)

    print("Build validation PASSED.")
    print(f"  Sources:   {result['num_sources']}")
    print(f"  Chunks:    {result['num_chunks']}")
    print(f"  Dimension: {result['dimension']}")
    print(f"  Model:     {result['model']}")


if __name__ == "__main__":
    main()