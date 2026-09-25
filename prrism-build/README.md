# prrism-build

Builds the PRRISM retrieval index from the **human-approved** part of the
[uai_lab Zotero group library](https://www.zotero.org/groups/5614036/uai_lab),
validates it, and publishes it as a GitHub Release for
[`prrism-search`](../prrism-search/README.md) to consume.

Builds run on the Vector Institute **Killarney** cluster, where embeddings are
served by a [`vec-inf`](https://github.com/VectorInstitute/vector-inference)
job hosting `bge-base-en-v1.5`. Publishing and fetching run locally.

```
Zotero (Benchmarks / Best Practices / Policy)
  -> zotero_loader      approved sources, deduped by item key
  -> build_index        extract text -> chunk -> embed -> tag -> FAISS
  -> out/builds/<id>/   candidate build
  -> quality gate + validate_build
  -> publish_build      atomic swap into out/current/
  -> release-prrism     GitHub Release "index-<id>"
  -> fetch-prrism / prrism-search runtime bootstrap
```

## What counts as "approved"

`zotero_loader.py` resolves three collections under the root collection
`HBREESE8`: **Benchmarks**, **Best Practices**, and **Policy**. It then walks
each of them *recursively*, so subcollections such as `Best Practices/Microsoft`
are included. `Candidate Sources` (where [prrism-watch](../prrism-watch/README.md)
files new items) is not under any of them, so unreviewed items never reach the
index.

An item filed in several approved collections becomes one source, keyed by its
Zotero item key (`source_id`), and keeps all of its collection memberships.
Sources are sorted by key, so the output order doesn't depend on the order the
Zotero API returns them in. The loader refuses to continue with an empty corpus.

To preview the approved corpus without building anything:

```bash
ZOTERO_API_KEY=... python prrism-build/zotero_loader.py
```

## How a build works (`build_index.py`)

1. **Text extraction.** Stored Zotero attachments (`imported_file`) are
   downloaded through the Zotero API and must be PDFs. Otherwise the source's
   URL is fetched, and PDFs are parsed with `pypdf` and HTML with `trafilatura`.
   If `trafilatura` isn't installed, the builder falls back to regex tag
   stripping. Failures are recorded in `failed_sources.json` rather than
   aborting the build.
2. **Chunking.** Chunks are 1200 characters with a 200-character overlap.
   Chunks shorter than 20 characters after cleaning are skipped and logged in
   `skipped_chunks.json`.
3. **Embedding.** Chunks are sent to the vec-inf `/embeddings` endpoint
   (OpenAI-compatible) in batches of 32. If a batch fails, the builder retries
   its chunks one by one, so a single bad chunk doesn't sink the batch.
4. **Tagging.** Each tag description in `include/tags.json` is embedded once.
   Every chunk then gets up to 2 practice, 2 ethics and 2 jurisdiction tags
   whose cosine similarity is **≥ 0.35**, stored as `{tag, score}`.
5. **Index.** A `faiss.IndexFlatIP` over L2-normalized vectors, so inner
   product equals cosine similarity. Row *i* of the index is `chunks.json[i]`.
6. **Quality gate.** At least **75%** of approved sources must produce at
   least one indexed chunk, or the build is rejected.
7. **Publish.** `publish_build()` validates the candidate, copies it into a
   staging directory next to `out/current/`, validates the copy again, and
   swaps it in with `os.replace`. If the swap fails, the previous build is
   restored. A rejected candidate stays in `out/builds/<build_id>/` for
   inspection, and `out/current/` is left untouched.

The build ID is a UTC timestamp (`20260924T180000Z`).

## Build artifacts

Every build directory must contain exactly these files:

| File | Contents |
|---|---|
| `faiss.index` | Vector index, one row per chunk |
| `chunks.json` | List of chunks: `source_id`, `url`, `title`, `chunk_id`, `text`, `collections`, `practice_tags`, `ethics_tags`, `jurisdiction_tags` |
| `source_summary.json` | Per-source stats, including `num_chunks` and `num_skipped_chunks` |
| `failed_sources.json` | Sources whose text couldn't be fetched or extracted |
| `skipped_chunks.json` | Chunks dropped before or during embedding |
| `build_meta.json` | `build_id`, `built_at`, `model`, `dim`, and source and chunk counts |

`validate_build.py` enforces:

- every file exists and loads;
- the FAISS vector count equals the chunk count;
- `build_meta.dim` matches the index dimension;
- the counts in `build_meta` agree with `chunks.json` and
  `source_summary.json`;
- every chunk has the required fields.

The same file list is hard-coded in `validate_build.py`, `publish_release.py`,
`fetch_release.py` and `prrism-search/runtime/runtime_artifacts.py`. If you add
or rename an artifact, update all four.

## Scripts

| Script | Where it runs | What it does |
|---|---|---|
| `rebuild-prrism` | Killarney | Loads modules and the venv, launches the vec-inf BGE job, polls until READY (up to 30 min), checks `/models`, runs `build_index.py`, and shuts the job down on exit. Needs `ZOTERO_API_KEY`. |
| `release-prrism [--dry-run]` | Local | Copies `out/current/` from Killarney with `scp` (host alias `killarney`), then `publish_release.py` validates it, packages it as `prrism-index-<id>.tar.gz` and runs `gh release create index-<id>`. `--dry-run` validates and packages the build but doesn't create a release. |
| `fetch-prrism [--tag index-…]` | Local | Downloads the newest (or a specific) published `index-*` release from `anatxalisha-a11y/PRRISM-v-9.26` with `gh`, verifies its SHA-256, extracts it (rejecting unexpected files or unsafe paths), validates it, and activates it into `out/current/`. |

`rebuild-prrism` hard-codes Killarney-specific values: the Slurm account
`aip-creager`, the module versions, and the venv and `vec-inf` paths under
`/project/6101846/aanat`. Edit them if the build moves to a different
account or user.

Release tooling uses the GitHub CLI (`gh`), which must be installed and
authenticated.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ZOTERO_API_KEY` | — | Read access to the group library (required) |
| `VEC_INF_URL` | `http://kn105:47870/v1` | Embedding server; `rebuild-prrism` sets this from `vec-inf status` |
| `VEC_INF_MODEL` | `bge-base-en-v1.5` | Model name sent to the server. It must match `EMBED_MODEL_NAME` in prrism-search. |
| `ZOTERO_LIBRARY_ID` | `5614036` | |
| `ZOTERO_LIBRARY_TYPE` | `group` | |
| `ZOTERO_ROOT_COLLECTION` | `HBREESE8` | Parent of the approved collections |

## Dependencies

There's no `requirements.txt` here yet. A build needs `faiss`, `numpy`,
`requests` and `pypdf`, and optionally `trafilatura` for better HTML
extraction. `zotero_loader.py` adds `../prrism-watch` to `sys.path` so it can
reuse `sourcing.zotero_client.ZoteroClient`, which means the prrism-watch
directory must be present.

## Tests

From the repo root (no network or keys needed):

```bash
python -m pytest prrism-build/tests -q
```
