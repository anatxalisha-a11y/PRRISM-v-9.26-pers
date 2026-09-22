# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository structure

Two largely independent layers:

- **Root (RAG search layer):** `search_engine.py` + `source.py` — local FAISS vector search over `out/chunks.json` / `out/faiss.index`, served via a Streamlit UI. The embedding model is `BAAI/bge-base-en-v1.5` (loaded once globally at import time). The index/chunk build script is **not in this repo** (see `prrism-watch/README.md` "Known TODOs").
- **`prrism-watch/` (sourcing pipeline):** monthly automation that discovers Canadian facial recognition primary sources and files them into the [uai_lab Zotero group library](https://www.zotero.org/groups/5614036/uai_lab) `Candidate Sources` collection for manual triage. Runs via GitHub Actions on the 1st of each month.

`include/tags.json` is the shared taxonomy (practice/ethics/jurisdiction tags) used by both layers.

## Commands

### Streamlit search UI (root layer)

```bash
pip install streamlit sentence-transformers faiss-cpu numpy
streamlit run source.py
```

Requires `out/faiss.index` and `out/chunks.json` to exist.

### prrism-watch pipeline

```bash
cd prrism-watch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY=...
export ZOTERO_API_KEY=...
export TAVILY_API_KEY=...

python -m sourcing.run_pipeline
```

### Tests (no network, no API keys needed)

```bash
cd prrism-watch
pip install pytest
python -m pytest tests/ -q
```

## prrism-watch pipeline architecture

```
discover (5 families) -> dedup -> keyword pre-filter -> Groq LLM judge -> insert into Zotero
```

- **Discover** (`sourcing/discover_*.py`): OpenAlex API (academic), curated RSS/scan_page/scan_index sources in `sourcing/sources.yaml` (government/municipal/company), Tavily search (news).
- **Dedup** (`sourcing/dedup.py`): checks `state/seen.json` (local, committed to repo) and `state/zotero_seen.json` (incremental mirror of Zotero collection tree). The Zotero state syncs only items changed since the last library version, not the full tree each run.
- **Filter** (`sourcing/filter_candidates.py`): keyword pre-pass first (government/municipal/company only require a facial-recognition term; news/academic also require a Canada jurisdiction term), then Groq (`openai/gpt-oss-120b` by default, overridable via `GROQ_MODEL` env var). The judge assigns `practice_tags`, `ethics_tags`, `jurisdiction_tags` from `include/tags.json`. Rate-limit resilience: 2.5s throttle between calls, exponential backoff on 429/5xx, `MAX_JUDGE_CALLS_PER_RUN=300` cap, circuit breaker after 3 consecutive failures. Candidates that hit the cap or fail are left **unmarked** in state so they're retried next run.
- **Insert** (`sourcing/zotero_client.py`, `sourcing/item_builder.py`): plain Zotero Web API calls, no MCP. Item type by family: `academic→journalArticle`, `news→newspaperArticle`, `government/municipal/company→webpage`.

## Key configuration (`sourcing/config.py`)

All overridable via env vars:

| Env var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Relevance/tagging judge (required) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Override if model is decommissioned |
| `ZOTERO_API_KEY` | — | Zotero write access (required) |
| `ZOTERO_TARGET_COLLECTION` | `CSZS3XH5` | Where new items land |
| `ZOTERO_DEDUP_ROOT_COLLECTION` | `HBREESE8` | Subtree checked for dedup |
| `TAVILY_API_KEY` | — | News discovery (required) |
| `WATCH_CONTACT_EMAIL` | unset | Sent in User-Agent / OpenAlex polite-pool |

## Known open items

- **Municipal coverage gap**: 7 sources (Vancouver, Calgary, Edmonton, Ottawa, Halifax eScribe-hosted pages) render client-side and were dropped from `sources.yaml`; needs Playwright to re-add.
- **Tag format**: `practice:`/`ethics:`/`jurisdiction:` prefixes in `item_builder.py` are a placeholder — verify against real Zotero library items and fix if the convention differs.
- **Index build script**: the chunking/embedding/FAISS build pipeline that produces `out/chunks.json` and `out/faiss.index` is not in this repo yet.
- **Failure notifications**: broken runs only surface via GitHub's own workflow-failure email.
- **Groq JSON schema**: `response_format: json_object` guarantees valid JSON but not the expected schema — watch first live run logs for `couldn't parse Groq response` / `missing expected keys` messages.
