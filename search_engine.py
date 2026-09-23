import os
import json
from typing import List, Dict
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from reranker import rerank

# Use the same embedding model that was used when building the FAISS index
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
CURRENT_DIR = OUT_DIR / "current"

INDEX_PATH = CURRENT_DIR / "faiss.index"
CHUNKS_PATH = CURRENT_DIR / "chunks.json"

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 25

# Load once globally so it doesn't reload on every query
_model = SentenceTransformer(EMBED_MODEL_NAME)


def embed(texts: List[str]) -> List[List[float]]:
    """
    Convert input texts into normalized embeddings using a local model.
    These embeddings must live in the same vector space as the embeddings
    used to build the FAISS index.
    """
    if not texts:
        return []

    embeddings = _model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def load_chunks(path: Path) -> List[Dict]:
    """
    Load chunk metadata/text from JSON.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_index(index_path: Path) -> faiss.Index:
    """
    Load a serialized FAISS index from disk.
    """
    return faiss.read_index(str(index_path))


def search(
    index: faiss.Index,
    chunks: List[Dict],
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
) -> List[Dict]:
    """
    Embed the query locally, retrieve candidate chunks from FAISS,
    rerank them with the cross-encoder, and return the top results.
    """
    if not query or not query.strip():
        return []

    # We need at least top_k candidates, but never request more
    # vectors than actually exist in the index.
    candidate_k = min(
        max(candidate_k, top_k),
        index.ntotal,
    )

    qv = np.array(embed([query.strip()]), dtype=np.float32)
    faiss.normalize_L2(qv)

    scores, ids = index.search(qv, candidate_k)

    candidates = []

    for retrieval_rank, (score, idx) in enumerate(
        zip(scores[0], ids[0]),
        start=1,
    ):
        if idx == -1:
            continue

        item = chunks[int(idx)]

        candidates.append({
            "score": float(score),
            "retrieval_score": float(score),
            "retrieval_rank": retrieval_rank,
            "search_type": "semantic_local",
            **item,
        })

    reranked = rerank(query, candidates)

    for reranker_rank, result in enumerate(reranked, start=1):
        result["reranker_rank"] = reranker_rank

    return reranked[:top_k]


def format_tag_list(tag_matches: List[Dict], show_score: bool = True) -> str:
    """
    Pretty-print tag matches for Streamlit display.
    """
    if not tag_matches:
        return "[]"
    if show_score:
        return "[" + ", ".join(f"{m['tag']} ({m['score']:.4f})" for m in tag_matches) + "]"
    return "[" + ", ".join(m['tag'] for m in tag_matches) + "]"


def check_server() -> str:
    """
    Kept only so poc2link1.py can still call it without breaking.
    This app no longer depends on vec-inf.
    """
    return f"Using local embedding model: {EMBED_MODEL_NAME}"


def load_artifacts():
    """
    Load the currently published FAISS index and chunks metadata.
    """

    if not INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"Missing FAISS index: {INDEX_PATH}"
        )

    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing chunks file: {CHUNKS_PATH}"
        )

    index = load_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            "Published artifact mismatch: "
            f"FAISS contains {index.ntotal} vectors but "
            f"chunks.json contains {len(chunks)} chunks."
        )

    return index, chunks