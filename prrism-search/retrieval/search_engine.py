import os
import json
from typing import List, Dict
from pathlib import Path
from functools import lru_cache

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from retrieval.diversity import select_diverse_results
from retrieval.reranker import rerank


# Use the same embedding model that was used when building the FAISS index.
EMBED_MODEL_NAME = os.environ.get(
    "EMBED_MODEL_NAME",
    "BAAI/bge-base-en-v1.5",
)

BASE_DIR = Path(__file__).resolve().parents[2]

OUT_DIR = BASE_DIR / "out"
CURRENT_DIR = OUT_DIR / "current"

INDEX_PATH = CURRENT_DIR / "faiss.index"
CHUNKS_PATH = CURRENT_DIR / "chunks.json"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 25
DEFAULT_MAX_PER_SOURCE = 2


# Load once globally so it doesn't reload on every query.
_model = None


def get_embedding_model():
    """
    Load the embedding model once and reuse it for subsequent queries.
    """

    global _model

    if _model is None:
        _model = SentenceTransformer(
            EMBED_MODEL_NAME
        )

    return _model


def embed(
    texts: List[str],
) -> List[List[float]]:
    """
    Embed text with the same normalized BGE model used by the index.
    """

    if not texts:
        return []

    embeddings = (
        get_embedding_model()
        .encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )

    return embeddings.tolist()


def load_chunks(
    path: Path,
) -> List[Dict]:
    """
    Load chunk metadata/text from JSON.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_index(
    index_path: Path,
) -> faiss.Index:
    """
    Load a serialized FAISS index from disk.
    """

    return faiss.read_index(
        str(index_path)
    )


def search(
    index: faiss.Index,
    chunks: List[Dict],
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    source_count: int | None = None,
) -> List[Dict]:
    """
    Search PRRISM's indexed evidence.

    The query is embedded locally with BGE, candidate chunks are
    retrieved from FAISS, and those candidates are reranked.

    By default, top_k retains its original meaning: the requested
    number of final chunks.

    When source_count is supplied, diversity selection also ensures
    that the final evidence represents up to that many unique sources.
    """

    if not query or not query.strip():
        return []

    if top_k <= 0:
        return []

    if (
        source_count is not None
        and source_count <= 0
    ):
        raise ValueError(
            "source_count must be greater than 0"
        )

    # A source-level request needs at least enough final result slots
    # to represent every requested source.
    if source_count is not None:
        top_k = max(
            top_k,
            source_count,
        )

    # We need at least top_k candidates, but never request more
    # vectors than actually exist in the index.
    candidate_k = min(
        max(
            candidate_k,
            top_k,
        ),
        index.ntotal,
    )

    qv = np.array(
        embed(
            [query.strip()]
        ),
        dtype=np.float32,
    )

    faiss.normalize_L2(qv)

    scores, ids = index.search(
        qv,
        candidate_k,
    )

    candidates = []

    for retrieval_rank, (
        score,
        idx,
    ) in enumerate(
        zip(
            scores[0],
            ids[0],
        ),
        start=1,
    ):

        if idx == -1:
            continue

        item = chunks[
            int(idx)
        ]

        candidates.append(
            {
                "score": float(score),
                "retrieval_score": float(
                    score
                ),
                "retrieval_rank": (
                    retrieval_rank
                ),
                "search_type": (
                    "semantic_local"
                ),
                **item,
            }
        )

    reranked = rerank(
        query,
        candidates,
    )

    for reranker_rank, result in enumerate(
        reranked,
        start=1,
    ):

        result[
            "reranker_rank"
        ] = reranker_rank

    return select_diverse_results(
        reranked,
        top_k=top_k,
        max_per_source=(
            DEFAULT_MAX_PER_SOURCE
        ),
        min_sources=source_count,
    )


def format_tag_list(
    tag_matches: List[Dict],
    show_score: bool = True,
) -> str:
    """
    Pretty-print tag matches for Streamlit display.
    """

    if not tag_matches:
        return "[]"

    if show_score:

        return (
            "["
            + ", ".join(
                f"{m['tag']} "
                f"({m['score']:.4f})"
                for m in tag_matches
            )
            + "]"
        )

    return (
        "["
        + ", ".join(
            m["tag"]
            for m in tag_matches
        )
        + "]"
    )


def check_server() -> str:
    """
    Kept only so poc2link1.py can still call it without breaking.

    This app no longer depends on vec-inf.
    """

    return (
        "Using local embedding model: "
        f"{EMBED_MODEL_NAME}"
    )


@lru_cache(maxsize=1)
def load_artifacts():
    """
    Load the currently published FAISS index and chunks metadata.
    """

    if not INDEX_PATH.is_file():

        raise FileNotFoundError(
            f"Missing FAISS index: "
            f"{INDEX_PATH}"
        )

    if not CHUNKS_PATH.is_file():

        raise FileNotFoundError(
            f"Missing chunks file: "
            f"{CHUNKS_PATH}"
        )

    index = load_index(
        INDEX_PATH
    )

    chunks = load_chunks(
        CHUNKS_PATH
    )

    if index.ntotal != len(
        chunks
    ):

        raise RuntimeError(
            "Published artifact mismatch: "
            f"FAISS contains "
            f"{index.ntotal} vectors but "
            f"chunks.json contains "
            f"{len(chunks)} chunks."
        )

    return index, chunks