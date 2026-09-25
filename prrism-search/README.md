# prrism-search

The researcher-facing side of PRRISM. A researcher asks a question about facial
recognition, picks how many sources to review (3–10), and gets back a
**ranked list of sources from the curated corpus**, each with an explanation
of what it contributes to the question.

The Streamlit entry point is **`source.py` at the repo root**. It adds
`prrism-search/` to `sys.path`, so the packages here import as `retrieval`,
`generation` and `runtime`.

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...
streamlit run source.py
```

## Query flow

```
question
  -> retrieval/search_engine.search()
       embed query (BGE, normalized)
       FAISS top 25 candidates
       retrieval/reranker.rerank()              bge-reranker-v2-m3
       retrieval/diversity.select_diverse_results()
  -> generation/answer_generator.build_evidence()     group chunks by source -> S1, S2, ...
  -> generation/answer_generator.generate_answer()    Groq explains each source
  -> validate_generated_answer()                      attach trusted title/URL/rank
  -> render_ranked_results() in source.py
```

### Retrieval (`retrieval/`)

- **Embedding.** The query is embedded locally with `sentence-transformers`
  (`BAAI/bge-base-en-v1.5`). This must be the same model that built the index;
  see [prrism-build](../prrism-build/README.md).
- **Candidates.** FAISS returns `candidate_k` (25) chunks. The number is raised
  to at least `top_k` and capped at the index size.
- **Reranking.** Candidates are rescored with the `BAAI/bge-reranker-v2-m3`
  cross-encoder. `retrieval_score`, `retrieval_rank`, `reranker_score` and
  `reranker_rank` are all kept on each result.
- **Diversity.** No more than 2 chunks per `source_id`. When the app passes
  `source_count`, a first pass takes the best chunk from each new source until
  that many sources are represented. A second pass adds further chunks from
  those same sources. Results come back in relevance order. If the candidate
  pool contains fewer unique sources than requested, the app returns fewer.

### Generation (`generation/answer_generator.py`)

The ranking is decided by retrieval, **not by the LLM**:

- `build_evidence()` groups passages by source and assigns citation IDs
  (`S1`, `S2`, …) in rank order. It passes along collection names and
  taxonomy labels as context, but not similarity scores.
- `generate_answer()` calls Groq (`temperature=0.2`, JSON mode). The prompt
  asks for one summary per source, in the given order, using only that
  source's passages and no outside knowledge.
- `validate_generated_answer()` rejects any response whose shape, field set,
  count or citation-ID order differs from the evidence. The public result is
  then assembled from **trusted** evidence metadata (rank, title, URL,
  `source_id`), and only the `summary` text comes from the model.

### Runtime bootstrap (`runtime/runtime_artifacts.py`)

When the app starts, `ensure_runtime_artifacts()` checks `out/current/` at the
repo root:

- If all six build artifacts exist, they're used as they are.
- Otherwise (for example on a fresh cloud deploy), it downloads the newest
  published `index-*` release from `anatxalisha-a11y/PRRISM-v-9.26` through the
  GitHub API using `PRRISM_GITHUB_TOKEN`. It verifies the SHA-256 digest,
  extracts the archive safely, runs `validate_build`, and activates the build
  with `publish_build`. Both functions are imported from `prrism-build/`.

The loaded index is cached with `@st.cache_resource`. If loading fails, the
app shows a generic error and stops.

The app writes `[MEMORY] <label>: <MB>` lines (peak RSS) to stdout at startup
and after the artifacts load, for tracking memory on the deployment host. The
KB→MB conversion assumes Linux, so the numbers are off on macOS.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Answer generation (required) |
| `PRRISM_ANSWER_MODEL` | `openai/gpt-oss-120b` | Groq model |
| `EMBED_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Query embedder; must match the index |
| `RERANKER_BACKEND` | `local` | `local` (in-process CrossEncoder) or `hosted` (Hugging Face TEI endpoint) |
| `HF_RERANKER_ENDPOINT` | — | TEI base URL; `/rerank` is appended |
| `HF_TOKEN` | — | Bearer token for the endpoint |
| `PRRISM_GITHUB_TOKEN` | — | Read-only token; only needed when `out/current/` is missing |

### Reranker failure behavior

- `RERANKER_BACKEND=hosted` with a missing endpoint or token **raises an
  error**. A misconfigured deploy fails visibly.
- A hosted-reranker **network or HTTP error** falls back to FAISS order. Those
  results carry `reranker_score=None` and `reranker_fallback=True`, and a
  warning is logged.
- Any other `RERANKER_BACKEND` value raises `ValueError`.

The local backend loads the cross-encoder on the first query, not at startup.

## Tests

From the repo root (no network or keys needed). Embedding and reranking are
mocked, and the generation tests cover only evidence building, prompt
construction and validation; the live Groq call isn't tested.

```bash
python -m pytest prrism-search -q
```
