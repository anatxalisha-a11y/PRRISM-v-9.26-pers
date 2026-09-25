# PRRISM

A research assistant over a curated corpus of **Canadian facial recognition**
policy, best-practice and benchmark sources. The corpus is managed in the
[uai_lab Zotero group library](https://www.zotero.org/groups/5614036/uai_lab).

The repo has three parts. They share data through Zotero and through the
build artifacts in `out/current/`:

```
prrism-watch   monthly GitHub Action: discover new sources -> Zotero "Candidate Sources"
                    |
                    |  manual review: move keepers into Benchmarks / Best Practices / Policy
                    v
prrism-build   on the Killarney cluster: approved Zotero items -> chunks + embeddings + FAISS
                    |  validated, published as a GitHub Release "index-<build_id>"
                    v
prrism-search  Streamlit app (source.py): retrieve -> rerank -> diversify -> LLM explains each source
```

| Part | README | Runs where |
|---|---|---|
| Sourcing | [prrism-watch/README.md](prrism-watch/README.md) | GitHub Actions, on the 1st of each month |
| Index build | [prrism-build/README.md](prrism-build/README.md) | Vector Killarney (Slurm and vec-inf) |
| Search app | [prrism-search/README.md](prrism-search/README.md) | Locally or on a cloud host |

`include/tags.json` holds the shared practice, ethics and jurisdiction
taxonomy. prrism-watch uses it to tag new candidates, and prrism-build uses it
to tag each chunk.

## Quick start (search app)

```bash
pip install -r requirements.txt
prrism-build/fetch-prrism          # pull the latest index into out/current/ (needs gh auth)
export GROQ_API_KEY=...
streamlit run source.py
```

## Tests

All three parts are tested from the repo root. Tests need no network access and
no API keys.

```bash
python -m pytest -q
```
