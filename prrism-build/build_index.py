import os
import json
import re
import urllib.request
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
import requests
from io import BytesIO
from pypdf import PdfReader

from zotero_loader import (
    Source,
    load_approved_sources,
    ZOTERO_LIBRARY_ID,
    ZOTERO_LIBRARY_TYPE,
)

from sourcing.zotero_client import ZoteroClient

from datetime import datetime, timezone
import shutil

from publish_build import publish_build

try:
    import trafilatura
except ImportError:
    trafilatura = None


# ----------------------------
# Config
# ----------------------------
VEC_INF_URL = os.environ.get("VEC_INF_URL", "http://kn105:47870/v1")
VEC_INF_MODEL = os.environ.get("VEC_INF_MODEL", "bge-base-en-v1.5")

from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
REPO_ROOT = BUILD_DIR.parent

TAGS_PATH = REPO_ROOT / "include" / "tags.json"
OUT_DIR = REPO_ROOT / "out"
BUILDS_DIR = OUT_DIR / "builds"

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200

TOP_K_PRACTICE_TAGS = 2
TOP_K_ETHICS_TAGS = 2
TOP_K_JURISDICTION_TAGS = 2

JURISDICTION_TAG_THRESHOLD = 0.35
PRACTICE_TAG_THRESHOLD = 0.35
ETHICS_TAG_THRESHOLD = 0.35

FETCH_TIMEOUT_S = 30
EMBED_TIMEOUT_S = 120
MIN_TEXT_WARN_CHARS = 200

EMBED_BATCH_SIZE = 32
MIN_EMBED_TEXT_CHARS = 20

MIN_SOURCE_SUCCESS_RATE = 0.75


# ----------------------------
# vec-inf / embedding client
# ----------------------------
def post_json(url: str, payload: dict, timeout_s: int = EMBED_TIMEOUT_S) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    url = f"{VEC_INF_URL}/embeddings"
    payload = {"model": VEC_INF_MODEL, "input": texts}
    out = post_json(url, payload)
    data = sorted(out["data"], key=lambda x: x["index"])
    return [row["embedding"] for row in data]


def embed_resilient(
    texts: List[str],
    batch_size: int = EMBED_BATCH_SIZE,
    label: str = "texts",
) -> Tuple[List[List[float]], List[int], List[Dict]]:
    """
    Returns:
        embeddings: embeddings for successful texts only
        kept_indices: original indices from `texts` that succeeded
        failures: list of {"index": i, "error": "..."}
    """
    embeddings: List[List[float]] = []
    kept_indices: List[int] = []
    failures: List[Dict] = []

    if not texts:
        return embeddings, kept_indices, failures

    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        end = start + len(batch) - 1
        print(f"  Embedding {label} batch {start}-{end} / {total - 1}")

        try:
            batch_embeddings = embed_batch(batch)
            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    f"Embedding API returned {len(batch_embeddings)} embeddings for {len(batch)} inputs."
                )

            embeddings.extend(batch_embeddings)
            kept_indices.extend(range(start, start + len(batch)))

        except Exception as e:
            print(f"    Batch failed: {repr(e)}")
            print("    Falling back to one-by-one embedding...")

            for offset, text in enumerate(batch):
                original_idx = start + offset
                try:
                    single_embedding = embed_batch([text])
                    if len(single_embedding) != 1:
                        raise ValueError(
                            f"Embedding API returned {len(single_embedding)} embeddings for 1 input."
                        )
                    embeddings.append(single_embedding[0])
                    kept_indices.append(original_idx)
                except Exception as e2:
                    print(f"      Skipping bad {label} item {original_idx}: {repr(e2)}")
                    failures.append({
                        "index": original_idx,
                        "error": repr(e2),
                    })

    return embeddings, kept_indices, failures


# ----------------------------
# Helpers
# ----------------------------
def l2_normalize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def cosine_from_normalized(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))



def normalize_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt).strip()


def clean_text_for_embedding(txt: str) -> str:
    if txt is None:
        return ""
    txt = txt.replace("\x00", " ")
    txt = normalize_text(txt)
    return txt


def is_valid_embedding_text(txt: str) -> bool:
    if not txt:
        return False
    if len(txt.strip()) < MIN_EMBED_TEXT_CHARS:
        return False
    return True




def load_tags(path) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        tags = json.load(f)

    required = {
        "practice_tags",
        "ethics_tags",
        "jurisdiction_tags",
    }

    missing = required - set(tags)

    if missing:
        raise ValueError(
            f"tags.json is missing required categories: {sorted(missing)}"
        )

    for category in required:
        if not isinstance(tags[category], dict):
            raise ValueError(
                f"'{category}' must map tag names to descriptions."
            )

    return tags

def validate_corpus_quality(
    total_sources: int,
    successful_sources: int,
    min_success_rate: float = MIN_SOURCE_SUCCESS_RATE,
) -> Dict:
    """
    Validate that enough of the approved corpus was successfully indexed.

    This is separate from artifact validation:
    - artifact validation checks internal consistency
    - corpus quality validation checks source coverage
    """
    if total_sources <= 0:
        raise ValueError("Cannot validate corpus quality with zero approved sources.")

    success_rate = successful_sources / total_sources

    result = {
        "total_sources": total_sources,
        "successful_sources": successful_sources,
        "failed_sources": total_sources - successful_sources,
        "success_rate": success_rate,
        "minimum_success_rate": min_success_rate,
    }

    if success_rate < min_success_rate:
        raise ValueError(
            "Corpus quality gate failed: "
            f"{successful_sources}/{total_sources} sources indexed "
            f"({success_rate:.1%}); minimum required is "
            f"{min_success_rate:.1%}."
        )

    return result

# ----------------------------
# Web -> plain text
# ----------------------------
def fetch_url(url: str, timeout_s: int = FETCH_TIMEOUT_S) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }

    r = requests.get(
        url,
        timeout=timeout_s,
        headers=headers,
        allow_redirects=True,
    )
    r.raise_for_status()
    return r.text

def pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Extract plain text from a PDF response.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()

        if text:
            pages.append(text)

    return "\n\n".join(pages)

def html_to_text(html: str, url: str = "") -> str:
    if trafilatura is not None:
        txt = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
        )
        if txt and txt.strip():
            return txt

    txt = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    txt = re.sub(r"(?is)<.*?>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        j = min(i + max_chars, n)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j == n:
            break
        i = max(0, j - overlap_chars)

    return chunks


# ----------------------------
# Tag embedding + assignment
# ----------------------------
def tag_value_to_text(tag_name: str, value) -> str:
    """
    Convert a taxonomy entry into text suitable for embedding.

    Simple taxonomy entries may be strings.
    Rich taxonomy entries may contain a definition, aliases,
    and child jurisdictions.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        parts = [tag_name]

        definition = value.get("definition")
        if definition:
            parts.append(definition)

        aliases = value.get("aliases", [])
        if aliases:
            parts.append("Aliases: " + ", ".join(aliases))

        jurisdictions = value.get("jurisdictions", [])
        if jurisdictions:
            parts.append("Jurisdictions: " + ", ".join(jurisdictions))

        return ". ".join(parts)

    raise TypeError(
        f"Unsupported taxonomy value for '{tag_name}': "
        f"{type(value).__name__}"
    )

def build_tag_embedding_matrix(tag_dict: Dict[str, str], label: str) -> Tuple[List[str], np.ndarray]:
    tag_names = list(tag_dict.keys())
    tag_descriptions = [
    clean_text_for_embedding(
        tag_value_to_text(name, tag_dict[name])
    )
    for name in tag_names
    ]

    bad_tag_indices = [i for i, txt in enumerate(tag_descriptions) if not is_valid_embedding_text(txt)]
    if bad_tag_indices:
        raise ValueError(f"Invalid tag descriptions for {label}: indices {bad_tag_indices}")

    tag_vecs, kept_indices, failures = embed_resilient(
        tag_descriptions,
        batch_size=EMBED_BATCH_SIZE,
        label=f"{label}_tags",
    )

    if failures:
        raise ValueError(f"Failed to embed some {label} tags: {failures}")

    if len(kept_indices) != len(tag_names):
        raise ValueError(
            f"Expected embeddings for all {label} tags, got {len(kept_indices)} / {len(tag_names)}"
        )

    tag_mat = l2_normalize_matrix(np.array(tag_vecs, dtype=np.float32))
    return tag_names, tag_mat


def assign_top_tags(
    chunk_vec: np.ndarray,
    tag_names: List[str],
    tag_mat: np.ndarray,
    top_k: int,
    threshold: float,
) -> List[Dict]:
    scored = []
    for i, tag_name in enumerate(tag_names):
        score = cosine_from_normalized(chunk_vec, tag_mat[i])
        if score >= threshold:
            scored.append({"tag": tag_name, "score": round(score, 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ----------------------------
# Source processing
# ----------------------------

def process_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch a source and extract clean text according to its content type.

    HTML -> trafilatura/html_to_text
    PDF  -> pypdf

    Returns:
        (clean_text, error_message)
    """
    try:
        response = requests.get(
            url,
            timeout=FETCH_TIMEOUT_S,
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        is_pdf = (
            "application/pdf" in content_type
            or response.content.startswith(b"%PDF")
        )

        if is_pdf:
            txt = pdf_to_text(response.content)
        else:
            txt = html_to_text(
                response.text,
                url=response.url,
            )

        txt = normalize_text(txt)

        if not txt:
            return None, "No extractable text"

        return txt, None

    except requests.exceptions.HTTPError as e:
        status = None
        if getattr(e, "response", None) is not None:
            status = e.response.status_code

        return None, f"HTTPError {status}: {e}"

    except requests.exceptions.RequestException as e:
        return None, f"RequestException: {e}"

    except Exception as e:
        return None, f"Unexpected error: {e}"

def process_source(
    source: Source,
    zotero: ZoteroClient,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract clean text from an approved PRRISM source.

    Stored Zotero PDF -> download through Zotero and parse with pypdf.
    Otherwise         -> use the source's external URL.
    """
    if source.attachment_key:
        try:
            file_bytes = zotero.download_attachment(
                source.attachment_key
            )

            is_pdf = (
                "application/pdf" in source.content_type.lower()
                or file_bytes.startswith(b"%PDF")
            )

            if not is_pdf:
                return (
                    None,
                    "Unsupported Zotero stored attachment "
                    f"content type: {source.content_type or 'unknown'}",
                )

            txt = pdf_to_text(file_bytes)
            txt = normalize_text(txt)

            if not txt:
                return None, "No extractable text from Zotero attachment"

            return txt, None

        except requests.exceptions.HTTPError as e:
            status = None
            if getattr(e, "response", None) is not None:
                status = e.response.status_code

            return (
                None,
                f"Zotero attachment HTTPError {status}: {e}",
            )

        except requests.exceptions.RequestException as e:
            return (
                None,
                f"Zotero attachment RequestException: {e}",
            )

        except Exception as e:
            return (
                None,
                f"Zotero attachment error: {e}",
            )

    if not source.url:
        return None, "Source has neither a URL nor a Zotero attachment"

    return process_url(source.url)

# ----------------------------
# Build corpus
# ----------------------------
def build_index(
    sources: List[Source],
    tags: Dict[str, Dict[str, str]],
    zotero: ZoteroClient,
) -> Dict:
    all_chunks: List[Dict] = []
    source_summary: Dict[str, Dict] = {}
    failed_sources: List[Dict] = []
    skipped_chunks: List[Dict] = []

    practice_tag_names, practice_tag_mat = build_tag_embedding_matrix(
    tags["practice_tags"], label="practice"
    )

    ethics_tag_names, ethics_tag_mat = build_tag_embedding_matrix(
        tags["ethics_tags"], label="ethics"
    )

    jurisdiction_tag_names, jurisdiction_tag_mat = build_tag_embedding_matrix(
        tags["jurisdiction_tags"], label="jurisdiction"
    )

    

    # First pass: fetch, clean, chunk
    for source in sources:
        url = source.url

        print(f"Fetching: {source.title}")

        if source.attachment_key:
            print(
                f"  Zotero attachment: "
                f"{source.attachment_key}"
            )
        else:
            print(f"  URL: {url}")

        txt, err = process_source(
            source,
            zotero,
        )

        if err is not None:
            print(f"  ERROR: {err}")
            failed_sources.append({
                "url": url,
                "error": err,
            })
            continue

        if txt is None:
            print("  ERROR: got no text back; skipping.")
            failed_sources.append({
                "url": url,
                "error": "No text returned",
            })
            continue

        if len(txt) < MIN_TEXT_WARN_CHARS:
            print(f"  WARN: extracted text looks very short ({len(txt)} chars). Keeping anyway.")

        raw_chunks = chunk_text(
            txt,
            max_chars=CHUNK_MAX_CHARS,
            overlap_chars=CHUNK_OVERLAP_CHARS,
        )
        print(f"  Extracted {len(txt)} chars -> {len(raw_chunks)} chunks")

        if not raw_chunks:
            print("  WARN: no chunks produced; skipping source.")
            failed_sources.append({
                "url": url,
                "error": "No chunks produced after extraction",
            })
            continue

        source_id = source.source_id

        kept_for_source = 0
        skipped_for_source = 0

        for chunk_id, ch in enumerate(raw_chunks):
            clean_chunk = clean_text_for_embedding(ch)

            if not is_valid_embedding_text(clean_chunk):
                skipped_for_source += 1
                skipped_chunks.append({
                    "source_id": source_id,
                    "url": url,
                    "chunk_id": chunk_id,
                    "reason": f"Chunk too short or invalid after cleaning (len={len(clean_chunk)})",
                    "text_preview": clean_chunk[:200],
                })
                continue

            all_chunks.append({
                "source_id": source_id,
                "url": url,
                "title": source.title,
                "collections": list(source.collections),
                "chunk_id": chunk_id,
                "text": clean_chunk,
            })
            kept_for_source += 1

        source_summary[source_id] = {
            "source_id": source_id,
            "url": url,
            "title": source.title,
            "item_type": source.item_type,
            "collections": list(source.collections),
            "num_raw_chunks": len(raw_chunks),
            "num_chunks": kept_for_source,
            "num_skipped_chunks": skipped_for_source,
            "practice_tags": {},
            "ethics_tags": {},
            "jurisdiction_tags": {},
        }

        if kept_for_source == 0:
            print("  WARN: all chunks for this source were invalid for embedding.")
            failed_sources.append({
                "url": url,
                "error": "All chunks invalid for embedding",
            })

    if not all_chunks:
        raise ValueError(
            "No embeddable chunks were created from the provided sources. "
            "All sources may have failed, been blocked, or produced invalid text."
        )

    # Second pass: embed all valid chunks
    texts_for_embed = [item["text"] for item in all_chunks]

    print(f"\nEmbedding {len(texts_for_embed)} chunks via {VEC_INF_URL}/embeddings ...")
    chunk_vecs_raw, kept_indices, embed_failures = embed_resilient(
        texts_for_embed,
        batch_size=EMBED_BATCH_SIZE,
        label="chunks",
    )

    if embed_failures:
        for failure in embed_failures:
            bad_i = failure["index"]
            bad_item = all_chunks[bad_i]
            skipped_chunks.append({
                "source_id": bad_item["source_id"],
                "url": bad_item["url"],
                "chunk_id": bad_item["chunk_id"],
                "reason": f"Embedding failure: {failure['error']}",
                "text_preview": bad_item["text"][:200],
            })

    if not kept_indices:
        raise ValueError("Embedding failed for all chunks; cannot build FAISS index.")

    kept_chunks = [all_chunks[i] for i in kept_indices]
    xb = l2_normalize_matrix(np.array(chunk_vecs_raw, dtype=np.float32))

    if len(kept_chunks) != xb.shape[0]:
        raise ValueError(
            f"Mismatch after embedding: {len(kept_chunks)} kept chunks vs {xb.shape[0]} embeddings"
        )

    # Reset source counts because some chunks may have failed during embedding
    for sid in source_summary:
        source_summary[sid]["num_chunks"] = 0
        source_summary[sid]["practice_tags"] = {}
        source_summary[sid]["ethics_tags"] = {}
        source_summary[sid]["jurisdiction_tags"] = {}

    # Third pass: auto-tag each chunk
    print("Assigning tags to chunks ...")
    final_chunks: List[Dict] = []

    for i, item in enumerate(kept_chunks):
        chunk_vec = xb[i]

        practice_matches = assign_top_tags(
            chunk_vec=chunk_vec,
            tag_names=practice_tag_names,
            tag_mat=practice_tag_mat,
            top_k=TOP_K_PRACTICE_TAGS,
            threshold=PRACTICE_TAG_THRESHOLD,
        )

        ethics_matches = assign_top_tags(
            chunk_vec=chunk_vec,
            tag_names=ethics_tag_names,
            tag_mat=ethics_tag_mat,
            top_k=TOP_K_ETHICS_TAGS,
            threshold=ETHICS_TAG_THRESHOLD,
        )

        jurisdiction_matches = assign_top_tags(
        chunk_vec=chunk_vec,
        tag_names=jurisdiction_tag_names,
        tag_mat=jurisdiction_tag_mat,
        top_k=TOP_K_JURISDICTION_TAGS,
        threshold=JURISDICTION_TAG_THRESHOLD,
        )

        enriched_item = dict(item)
        enriched_item["practice_tags"] = practice_matches
        enriched_item["ethics_tags"] = ethics_matches
        enriched_item["jurisdiction_tags"] = jurisdiction_matches
        final_chunks.append(enriched_item)

        sid = enriched_item["source_id"]
        source_summary[sid]["num_chunks"] += 1

        for match in practice_matches:
            tag = match["tag"]
            source_summary[sid]["practice_tags"][tag] = (
                source_summary[sid]["practice_tags"].get(tag, 0) + 1
            )

        for match in ethics_matches:
            tag = match["tag"]
            source_summary[sid]["ethics_tags"][tag] = (
                source_summary[sid]["ethics_tags"].get(tag, 0) + 1
            )
        
        for match in jurisdiction_matches:
            tag = match["tag"]
            source_summary[sid]["jurisdiction_tags"][tag] = (
                source_summary[sid]["jurisdiction_tags"].get(tag, 0) + 1
            )

    # Convert source-level tag counts into sorted lists
    for sid, rec in source_summary.items():
        practice_sorted = sorted(
            rec["practice_tags"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        ethics_sorted = sorted(
            rec["ethics_tags"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        jurisdiction_sorted = sorted(
            rec["jurisdiction_tags"].items(),
            key=lambda x: x[1],
            reverse=True,
        )

        rec["practice_tags"] = [
            {"tag": tag, "count": count}
            for tag, count in practice_sorted
        ]
        rec["ethics_tags"] = [
            {"tag": tag, "count": count}
            for tag, count in ethics_sorted
        ]
        rec["jurisdiction_tags"] = [
            {"tag": tag, "count": count}
            for tag, count in jurisdiction_sorted
        ]

    # Fourth pass: build FAISS index
    dim = xb.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(xb)

    return {
        "model": VEC_INF_MODEL,
        "dim": dim,
        "chunks": final_chunks,
        "source_summary": source_summary,
        "failed_sources": failed_sources,
        "skipped_chunks": skipped_chunks,
        "faiss_index": faiss_index,
    }


# ----------------------------
# Save outputs
# ----------------------------
def save_outputs(
    index: Dict,
    out_dir: str = OUT_DIR,
    *,
    build_id: str,
    approved_sources: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    faiss.write_index(index["faiss_index"], os.path.join(out_dir, "faiss.index"))

    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(index["chunks"], f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "source_summary.json"), "w", encoding="utf-8") as f:
        json.dump(index["source_summary"], f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "failed_sources.json"), "w", encoding="utf-8") as f:
        json.dump(index.get("failed_sources", []), f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "skipped_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(index.get("skipped_chunks", []), f, ensure_ascii=False, indent=2)

    successfully_indexed_sources = sum(
        1
        for source in index["source_summary"].values()
        if source["num_chunks"] > 0
    )

    build_meta = {
        "build_id": build_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "model": index["model"],
        "dim": index["dim"],
        "approved_sources": approved_sources,
        "successfully_indexed_sources": successfully_indexed_sources,
        "sources_without_final_chunks": (
            approved_sources - successfully_indexed_sources
        ),
        "recorded_source_failure_events": len(
            index.get("failed_sources", [])
        ),
        "final_chunks": len(index["chunks"]),
        "skipped_chunk_events": len(
            index.get("skipped_chunks", [])
        ),
    }

    with open(
        os.path.join(out_dir, "build_meta.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            build_meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nWrote:")
    print(f"  {out_dir}/faiss.index")
    print(f"  {out_dir}/chunks.json")
    print(f"  {out_dir}/source_summary.json")
    print(f"  {out_dir}/failed_sources.json")
    print(f"  {out_dir}/skipped_chunks.json")
    print(f"  {out_dir}/build_meta.json")


def create_candidate_dir() -> Path:
    """
    Create a unique directory for a candidate PRRISM build.
    """

    build_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    candidate_dir = BUILDS_DIR / build_id

    # Extremely unlikely, but prevents accidental reuse if two
    # builds begin during the same second.
    if candidate_dir.exists():
        raise RuntimeError(
            f"Build directory already exists: {candidate_dir}"
        )

    candidate_dir.mkdir(parents=True)

    return candidate_dir

# ----------------------------
# Main
# ----------------------------
def main():
    try:
        _ = urllib.request.urlopen(
            f"{VEC_INF_URL}/models",
            timeout=5,
        ).read().decode("utf-8")

        print("Embedding server reachable. /models OK.")

    except Exception as e:
        print("ERROR: cannot reach embedding server at:", VEC_INF_URL)
        print(" ", repr(e))
        return

    try:
        api_key = os.environ.get("ZOTERO_API_KEY")

        if not api_key:
            raise RuntimeError("ZOTERO_API_KEY is not set.")

        zotero = ZoteroClient(
            api_key=api_key,
            library_id=ZOTERO_LIBRARY_ID,
            library_type=ZOTERO_LIBRARY_TYPE,
        )

        print("\nLoading approved PRRISM corpus from Zotero...")

        sources = load_approved_sources(zotero)

        print(f"Loaded {len(sources)} approved sources.")

        tags = load_tags(TAGS_PATH)

        print(
            f"Loaded {len(tags['practice_tags'])} practice tags, "
            f"{len(tags['ethics_tags'])} ethics tags, and "
            f"{len(tags['jurisdiction_tags'])} jurisdiction tags."
        )

        index = build_index(
            sources,
            tags,
            zotero,
        )

        candidate_dir = create_candidate_dir()

        print(f"\nWriting candidate build to:")
        print(f"  {candidate_dir}")

        try:
            save_outputs(
                index,
                candidate_dir,
                build_id=candidate_dir.name,
                approved_sources=len(sources),
            )

            print("\nChecking corpus quality...")

            successful_sources = sum(
                1
                for source in index["source_summary"].values()
                if source["num_chunks"] > 0
            )

            quality = validate_corpus_quality(
                total_sources=len(sources),
                successful_sources=successful_sources,
            )

            print(
                "Corpus quality gate passed: "
                f"{quality['successful_sources']}/{quality['total_sources']} "
                f"sources indexed ({quality['success_rate']:.1%})."
            )

            print("\nValidating and publishing candidate...")

            validation = publish_build(
                candidate_dir=candidate_dir,
                out_dir=OUT_DIR,
            )
        except Exception:
            print("\nCandidate build was NOT published.")
            print(f"Candidate retained for inspection:")
            print(f"  {candidate_dir}")
            raise

        print("\nBuild validated and published successfully.")
        print(f"Active build:")
        print(f"  {OUT_DIR / 'current'}")

        print(
            f"Validated {validation['num_sources']} sources, "
            f"{validation['num_chunks']} chunks, "
            f"dimension {validation['dimension']}."
        )

        
        print(f"Successful sources: {len(index['source_summary'])}")
        print(
            f"Failed sources: "
            f"{len(index.get('failed_sources', []))}"
        )
        print(f"Final embedded chunks: {len(index['chunks'])}")
        print(
            f"Skipped chunks: "
            f"{len(index.get('skipped_chunks', []))}"
        )

        if index.get("failed_sources"):
            print("\nFailed source summary:")

            for item in index["failed_sources"]:
                print(f"  - {item['url']}")
                print(f"    {item['error']}")

    except Exception as e:
        print("\nFATAL ERROR during build:")
        print(f"  {repr(e)}")
        raise


if __name__ == "__main__":
    main()