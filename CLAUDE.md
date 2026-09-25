# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository structure

PRRISM is a RAG search tool over a curated Zotero corpus of Canadian facial recognition sources. Three layers share a data flow through the [uai_lab Zotero group library](https://www.zotero.org/groups/5614036/uai_lab) and the `out/current/` build artifacts:

```
prrism-watch  --(auto-sourced candidates)-->  Zotero "Candidate Sources"
                                                  | manual triage into Benchmarks / Best Practices / Policy
prrism-build  <--(approved collections)---------+
     | builds + validates out/current/, publishes as GitHub release "index-*"
prrism-search <--(local out/current/, or downloads latest release at startup)
```

- **`prrism-watch/`**: monthly sourcing pipeline (GitHub Actions, 1st of each month). It files new candidates into the `Candidate Sources` collection. See `prrism-watch/README.md` for the full design notes.
- **`prrism-build/`**: builds the index from the approved Zotero collections (`Benchmarks`, `Best Practices`, `Policy` under root `HBREESE8`). It runs on the Vector Institute Killarney Slurm cluster, where embeddings come from a `vec-inf`-hosted `bge-base-en-v1.5` server (`VEC_INF_URL`).
- **`prrism-search/`**: the `retrieval/`, `generation/` and `runtime/` packages for the Streamlit app. The app entry point is **`source.py` at the repo root**, not inside this directory.

Each layer has a README with the full details (`prrism-watch/README.md`, `prrism-build/README.md`, `prrism-search/README.md`). Read the relevant one before making non-trivial changes to a layer.

`include/tags.json` is the shared practice/ethics/jurisdiction taxonomy. prrism-watch uses it for LLM tagging and prrism-build uses it for embedding-similarity tag assignment on chunks.

### Cross-layer imports (non-obvious)

- `prrism-build/` and `prrism-search/` are hyphenated, so they aren't importable packages. Modules add paths to `sys.path` themselves:
  - `prrism-build/zotero_loader.py` adds `prrism-watch/` so it can reuse `sourcing.zotero_client.ZoteroClient`.
  - `prrism-search/runtime/runtime_artifacts.py` adds `prrism-build/` so it can reuse `publish_build` / `validate_build` when it bootstraps.
  - prrism-build tests load `build_index.py` via `importlib` after inserting `prrism-build/` into `sys.path`.
- `prrism-search` modules import each other as top-level packages (`from retrieval.search_engine import ...`). This works because `source.py` inserts `prrism-search/` into `sys.path` and root `pytest.ini` puts it on `pythonpath`.
- `search_engine.py` and `runtime_artifacts.py` locate the repo root with `Path(__file__).resolve().parents[2]`. Update that if either file moves.

## Commands

### Tests (all layers, no network or API keys needed)

Run these from the repo root, since `pytest.ini` sets `pythonpath` and `testpaths`:

```bash
python -m pytest -q                                              # everything
python -m pytest prrism-search/retrieval/tests -q                # one directory
python -m pytest prrism-build/tests/test_validate_build.py::test_name -q   # single test
```

### Search app

```bash
pip install -r requirements.txt
streamlit run source.py            # from the repo root
```

### Index build / release (prrism-build)

- `prrism-build/rebuild-prrism`: runs on Killarney. It loads modules, launches the vec-inf BGE job, waits for READY, runs `build_index.py`, and shuts the job down on exit. Requires `ZOTERO_API_KEY`.
- `prrism-build/release-prrism`: runs locally. It uses `scp` to copy `out/current` from Killarney, then `publish_release.py` validates it and publishes it as a GitHub release.
- `prrism-build/fetch-prrism`: runs locally. `fetch_release.py` downloads the latest `index-*` release with `gh` and publishes it into `out/`.

### prrism-watch pipeline

```bash
cd prrism-watch
pip install -r requirements.txt
export GROQ_API_KEY=... ZOTERO_API_KEY=... TAVILY_API_KEY=...
python -m sourcing.run_pipeline
```

## Build artifacts and publishing contract

A valid build directory contains exactly these files, and several modules check them independently: `faiss.index`, `chunks.json`, `source_summary.json`, `failed_sources.json`, `skipped_chunks.json`, `build_meta.json`. The same set is duplicated in `validate_build.py`, `publish_release.py`, `fetch_release.py` and `runtime_artifacts.py`, so change all four together. Each chunk requires `source_id` (the Zotero item key), `url`, `title`, `chunk_id`, `text`, and `practice_tags`/`ethics_tags`/`jurisdiction_tags` (lists of `{tag, score}`).

- `build_index.py` writes a candidate to `out/builds/<build_id>/` and runs a corpus-quality gate. `publish_build.publish_build()` then validates the candidate, stages it next to `out/current/`, and swaps it in atomically with `os.replace` (the previous build is restored on failure). A failed candidate is left on disk for inspection. **Never write directly into `out/current/`.**
- Releases are tagged `index-*` and carry a `prrism-index-*.tar.gz` asset on `anatxalisha-a11y/PRRISM-v-9.26`. Their digests are verified on download.
- `out/` is gitignored.

## prrism-search runtime

Each query goes through: BGE query embedding (local `sentence-transformers`; it must match the build model) → FAISS top `candidate_k` (25) → rerank (`bge-reranker-v2-m3`) → `diversity.select_diverse_results` (at most 2 chunks per `source_id`, with optional `min_sources`) → `answer_generator` groups the chunks by source and calls Groq.

The LLM **does not** choose source rank, identity, title, or URL. Rank comes from retrieval. The model only explains each source's contribution, and `validate_generated_answer` checks its output. Similarity scores are hidden from the model and the UI.

At startup, `ensure_runtime_artifacts()` keeps a complete local `out/current/` if one exists. Otherwise it downloads the newest release using `PRRISM_GITHUB_TOKEN`, for cloud deployment.

| Env var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Answer generation (required) |
| `PRRISM_ANSWER_MODEL` | `openai/gpt-oss-120b` | Groq model for answers |
| `RERANKER_BACKEND` | `local` | `local` (CrossEncoder) or `hosted` (HF TEI endpoint) |
| `HF_RERANKER_ENDPOINT`, `HF_TOKEN` | — | Required when `RERANKER_BACKEND=hosted` (a missing value raises an error instead of falling back silently) |
| `EMBED_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Query embedder; must match the index |
| `PRRISM_GITHUB_TOKEN` | — | Read-only token for release bootstrap |

prrism-build env vars: `VEC_INF_URL`, `VEC_INF_MODEL`, `ZOTERO_API_KEY`, `ZOTERO_LIBRARY_ID` (`5614036`), `ZOTERO_LIBRARY_TYPE` (`group`), `ZOTERO_ROOT_COLLECTION` (`HBREESE8`).

## prrism-watch pipeline architecture

```
discover (5 families) -> dedup -> keyword pre-filter -> Groq LLM judge -> insert into Zotero
```

- **Discover** (`sourcing/discover_*.py`): the OpenAlex API (academic); curated RSS, `scan_page` and `scan_index` sources in `sourcing/sources.yaml` (government, municipal and company); and Tavily search (news).
- **Dedup** (`sourcing/dedup.py`): checks `state/seen.json` and `state/zotero_seen.json`. Both files are committed back to the repo by the Action. The Zotero mirror syncs incrementally, fetching only items changed since the last library version.
- **Filter** (`sourcing/filter_candidates.py`): a keyword pre-pass comes first. Government, municipal and company candidates only need a facial-recognition term; news and academic candidates also need a Canada term. Candidates that pass go to the Groq judge (`GROQ_MODEL`, default `openai/gpt-oss-120b`), which assigns taxonomy tags. Throttling and limits: 2.5s between calls, backoff on 429/5xx, `MAX_JUDGE_CALLS_PER_RUN=300`, and a circuit breaker after 3 consecutive failures. Candidates that are capped or fail stay **unmarked** so the next run retries them.
- **Insert** (`sourcing/zotero_client.py`, `sourcing/item_builder.py`): plain Zotero Web API calls, no MCP. Item types: academic→`journalArticle`, news→`newspaperArticle`, everything else→`webpage`.

Config lives in `sourcing/config.py`. It reads `GROQ_API_KEY`, `GROQ_MODEL`, `ZOTERO_API_KEY`, `ZOTERO_TARGET_COLLECTION` (`CSZS3XH5`), `ZOTERO_DEDUP_ROOT_COLLECTION` (`HBREESE8`), `TAVILY_API_KEY` and `WATCH_CONTACT_EMAIL`.

## Known open items

- **Municipal coverage gap**: 7 client-rendered sources (the eScribe-hosted pages for Vancouver, Calgary, Edmonton, Ottawa and Halifax) were dropped from `sources.yaml`. Re-adding them needs Playwright.
- **Tag format**: the `practice:`/`ethics:`/`jurisdiction:` prefixes in `item_builder.py` are placeholders and still need to be checked against real library items.
- **Groq JSON schema**: `response_format: json_object` guarantees valid JSON, but not the expected schema. Watch for `couldn't parse Groq response` / `missing expected keys` in the logs.
- **Failure notifications**: broken watch runs only surface through GitHub's workflow-failure email.
