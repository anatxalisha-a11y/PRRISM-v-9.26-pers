# prrism-watch

Monthly automation that finds new **Canadian facial recognition primary
sources** — government, municipal, company, news, and academic material —
filters it for relevance, and files it into the `Candidate Sources`
collection of the [uai_lab Zotero group
library](https://www.zotero.org/groups/5614036/uai_lab) for manual review.
This is the sourcing/ingestion layer upstream of the rest of `prrism-rag`
(which reads from Zotero to build the search index) — it does not touch
`out/chunks.json` or `out/faiss.index` itself.

## Pipeline

```
discover (5 source families) -> dedup (local state + live Zotero check)
  -> filter (keyword pre-pass -> Groq relevance judge + tag assignment)
  -> insert into Zotero "Candidate Sources", tagged
```

1. **Discover** (`sourcing/discover_academic.py`, `discover_curated.py`,
   `discover_news.py`) — five source families:
   - **academic**: [OpenAlex](https://openalex.org) API, keyless, queried with
     a rotating set of terms. Returns clean title/authors/DOI/journal/abstract.
   - **government** / **municipal**: `sourcing/sources.yaml`'s curated list.
     Each entry is RSS-backed, `scan_page`-backed (one-hop: fetch once, links
     whose anchor text mentions facial recognition/biometrics become
     candidates — fits blogs/newsrooms), or `scan_index`-backed (two-hop: for
     meeting/agenda listings where link text is just a date — see "Curated
     source research" below for why that distinction exists).
   - **company**: same curated mechanism as government/municipal, different
     list in `sources.yaml`.
   - **news**: [Tavily](https://tavily.com), with a query set in
     `sources.yaml` written to favor primary sources (official releases,
     court text, regulator investigations) over secondary commentary. See
     "Tavily credit budget" below for exact per-run cost.

2. **Dedup** (`sourcing/dedup.py`) — before anything is judged or inserted,
   candidates are checked against:
   - the local `state/seen.json` (committed by the Action each run — covers
     both inserted items *and* previously rejected ones, so nothing is
     re-judged every month), and
   - `state/zotero_seen.json`, a persisted mirror of which Zotero items sit
     inside the collection tree rooted at `ZOTERO_DEDUP_ROOT_COLLECTION` (the
     parent of Candidate Sources, Benchmarks, Best Practices, and Policy), so
     an item already promoted out of Candidate Sources into a sibling
     collection still counts as seen. This mirror is kept incrementally, not
     rebuilt from scratch — Zotero's version-based sync means each run only
     fetches items that changed since the last recorded library version
     (`sync_zotero_seen_state()`), so cost stays flat as the library grows
     instead of re-walking every item in the tree every month.

3. **Filter** (`sourcing/filter_candidates.py`) — two stages:
   - a cheap keyword pre-pass (must mention facial recognition; news/academic
     candidates additionally need a Canada jurisdiction term, reusing the
     alias/jurisdiction lists already in `include/tags.json` — government/
     municipal/company candidates skip that check since they're already
     Canada-scoped by curation),
   - then [Groq](https://groq.com) (llama-3.3-70b-versatile, free tier — 30
     RPM / 1,000 requests/day, well over what a monthly batch needs) judges
     whether this is a genuine primary source and assigns `practice_tags` /
     `ethics_tags` / `jurisdiction_tags` from the existing taxonomy in
     `include/tags.json`. The exact required JSON shape is spelled out in
     the prompt itself (Groq's `response_format: json_object` guarantees
     valid JSON but not a specific schema the way Gemini's `responseSchema`
     did — see "Resilience" below for why Gemini was replaced), and the
     parsed result is validated defensively rather than trusted to match.
     The judge also gets a sample of tags already in use elsewhere in the
     Zotero library (`ZoteroClient.get_library_tags()`) as reference, so new
     items lean toward the same tagging style already established rather
     than the model inventing its own — see the tag-format note under
     Insert below.

4. **Insert** (`sourcing/zotero_client.py`, `item_builder.py`) — plain calls to
   the Zotero Web API (no MCP: a GitHub Actions run is a fresh, disposable
   container each time, with nothing to keep an MCP server session alive for).
   Items are typed `journalArticle` (academic), `newspaperArticle` (news), or
   `webpage` (government/municipal/company), filed into `Candidate Sources`,
   and tagged with `source-family:*`, `auto-sourced`, plus the assigned
   `practice:*` / `ethics:*` / `jurisdiction:*` tags.

   **Open question:** that `practice:`/`ethics:`/`jurisdiction:` prefix is a
   placeholder — invented before ever seeing how tags are actually formatted
   on the existing Benchmarks/Best Practices/Policy items, which is exactly
   the convention new items should match to minimize manual relabeling later.
   Check a few existing items' tags in the Zotero web UI and update
   `item_builder.py`'s tag-building block (and this note) to match.

## Zotero setup

Everything targets the **uai_lab** group library (id `5614036`):

| Collection | Key | Role |
|---|---|---|
| [Candidate Sources](https://www.zotero.org/groups/5614036/uai_lab/collections/CSZS3XH5/collection) | `CSZS3XH5` | Where the pipeline inserts new items, for manual triage |
| [parent collection](https://www.zotero.org/groups/5614036/uai_lab/collections/HBREESE8/collection) (siblings: Benchmarks, Best Practices, Policy, Candidate Sources) | `HBREESE8` | Dedup is checked against this whole subtree, not just Candidate Sources |

Both keys default in `sourcing/config.py` and can be overridden via env vars
(`ZOTERO_TARGET_COLLECTION`, `ZOTERO_DEDUP_ROOT_COLLECTION`) if the library
structure changes.

## Secrets (GitHub Actions repo secrets)

| Secret | Used for |
|---|---|
| `GROQ_API_KEY` | the relevance/tagging judge in `filter_candidates.py` — free key from [console.groq.com/keys](https://console.groq.com/keys), no card needed |
| `ZOTERO_API_KEY` | a Zotero API key ([zotero.org/settings/keys](https://www.zotero.org/settings/keys)) scoped to **write access on the uai_lab group only** |
| `TAVILY_API_KEY` | the news-discovery search API — free key from [tavily.com](https://tavily.com), no card needed |

Groq and Tavily over the more obvious Anthropic/Google CSE: both are free
with no card, and Google's Custom Search JSON API stopped accepting new
signups in Jan 2026 (full shutdown Jan 2027) so isn't provisionable for a
new setup at all. Neither swap needed a new dependency — both are called
over plain REST via `requests`. Groq itself replaced an initial Gemini
choice — see "Resilience" below for why.

One more, optional, not a secret: `WATCH_CONTACT_EMAIL` — if set, it's sent
to third parties as OpenAlex's "polite pool" `mailto` param (faster rate
limits) and in the `User-Agent` header on every scrape request to
government/municipal/company sites, so a site admin who notices the traffic
has a way to reach out. Unset by default; if you set it, use a project/team
address rather than a personal one, since it's genuinely sent outbound to
dozens of third-party services on every run.

## Running locally

```bash
cd prrism-watch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GROQ_API_KEY=...
export ZOTERO_API_KEY=...
export TAVILY_API_KEY=...

python -m sourcing.run_pipeline
```

Unit tests (no network, no API keys needed):

```bash
pip install pytest
python -m pytest tests/ -q
```

## Curated source research (government / municipal / company)

`sourcing/sources.yaml` is the curated seed list, built from a live research
pass (every URL/feed fetched and verified, not guessed) rather than from
memory. Findings:

**Government (federal + provincial/territorial)** — well covered by real
feeds. OPC Canada, BC's OIPC, Ontario's IPC (via its Decisia decisions
platform), Alberta's and Saskatchewan's OIPCs, and TBS/Justice (via canada.ca's
`io.canada.ca` news API) all have confirmed, working RSS/Atom feeds — these
are the highest-signal, lowest-maintenance sources in the whole list. The
rest (Manitoba, Quebec, New Brunswick, Nova Scotia, PEI, NL, Yukon, NWT,
Nunavut, ISED itself) have no feed and are on `scan_page`. Two specific
caveats: ISED's exact canada.ca department code for a filtered news feed
couldn't be confirmed (using the general dept news page instead), and NWT's
`atipp-nt.ca/feed/` is real but was returning zero items at research time —
worth revisiting periodically in case it starts populating.

**Municipal** — no city in the researched set exposes a public API or feed
for new council/police-board agenda items. Meeting-list pages also only show
dates as link text ("March 4, 2025"), never the actual agenda item title —
those only exist on each meeting's own page, frequently as a **PDF**, not
HTML (confirmed live against Toronto Police Services Board's WordPress site,
which serves agendas as `wp-content/uploads/.../Public-Meeting-Agenda-*.pdf`).
`discover_curated.py`'s `scan_index` mode handles that with two hops — pull
detail/document links off the listing page, fetch a bounded number of them,
extract PDF text where relevant (`pypdf`), and keyword-match the full text.

That got 6 of the original 15 sources genuinely working (verified: each one
fetches real detail-page content and keyword-matches it, not just "runs
without error"). The other 9 don't currently work, for two different
reasons, and both are marked directly in `sources.yaml` rather than left in
silently: **2 are blocked** (Toronto's TMMIS returns 403; Winnipeg's DMIS
fails TLS verification), and **7 were dropped entirely** — Vancouver's own
council page and all 4 eScribe-hosted cities' council/police-board pages
(Calgary, Edmonton, Ottawa, Halifax) render their meeting list client-side
via ASP.NET postback (`javascript:__doPostBack(...)`) rather than serving it
as real HTML, so a plain HTTP fetch sees an essentially empty page — verified
directly by fetching Ottawa's eScribe page, which has 5 links total and none
of them point anywhere useful. That's not a tunable `detail_link_pattern`
problem; it needs a headless browser (Playwright) to actually render the
page, which is a real scope decision (new dependency, browser binaries in
CI, slower/more fragile runs) — deliberately left out for now rather than
silently shipped as broken. Re-add those 7 if that gets built. City
open-data portals (a mix of CKAN, Socrata, and Esri ArcGIS Hub depending on
the city) host structured datasets, not agenda text, so they aren't useful
as a discovery feed here regardless.

**Company** — AIDA (the Artificial Intelligence and Data Act, part of Bill
C-27) died at prorogation in January 2025 and has not been replaced by a
comprehensive federal AI statute as of this research (Bill C-36, tabled June
2026, covers privacy/deepfake provisions but not AI governance broadly). That
makes ISED's [voluntary AI code of
conduct](https://ised-isde.canada.ca/site/ised/en/voluntary-code-conduct-responsible-development-and-management-advanced-generative-ai-systems)
signatory list (49 organizations, fetched live) the only primary-source
federal AI-governance artifact with a company list right now — worth
re-checking if AI-specific legislation reappears. `sources.yaml`'s `company`
section includes signatories with confirmed responsible-AI content plus
facial-recognition/biometrics vendors with a Canadian nexus; Clearview AI,
NEC, and Idemia have the strongest documented ties (a formal OPC finding
against Clearview, named/dollar-figured police contracts for NEC and Idemia —
including Ottawa's live June 2026 Idemia contract). Cognitec, iProov, and
Paravision are included but their Canadian nexus is unconfirmed. Global
newsrooms with no Canada-specific or FR-adjacent content (IBM, SAP,
Salesforce) were deliberately left out to avoid flooding the LLM judge with
noise from unrelated general AI content.

## Tavily credit budget

Tavily bills **1 credit per search call** at `search_depth: "basic"` (what
`search_client.py` always uses) — `max_results` doesn't change the cost, and
it's independent of how many results actually come back. `discover_news.py`
runs exactly one search per entry in `sources.yaml`'s `news_queries` (4 by
default), so:

```
credits per run = number of news_queries  (currently 4 x 1 credit = 4 credits)
credits per month = credits per run x scheduled runs per month  (1 monthly cron run = 4 credits/month)
```

Against a 1,500-credit/month free trial, that's **~0.3% of budget per
scheduled run** — even a dozen extra manual `workflow_dispatch` test runs in
a month (48 credits) barely registers. `MAX_QUERIES_PER_RUN = 50` in
`discover_news.py` is a defensive ceiling, not a real constraint at current
usage — it exists so that if `news_queries` ever grows substantially (e.g.
someone pastes in a much longer query list later), a run gets capped and
logs a warning instead of silently spending way more than expected. Every
run also logs its actual credit usage (`[discover_news] used ~N Tavily
credits this run`) so spend is visible in the Action's log without doing the
math by hand.

## Live dry-run results (discovery + keyword pre-filter only, no LLM/Zotero)

Run against the full curated list on 2026-08-30 (government/company), with
municipal re-run per-source on 2026-08-31 after the finding below:

| Family | Raw candidates | Pass keyword pre-filter |
|---|---|---|
| government | 1,333 | 20 |
| municipal | 2 (from 6 working sources; see below) | 0 |
| company | 398 | 23 |

Bugs this surfaced and fixed: a Cloudflare email-obfuscation link
(`cdn-cgi/l/email-protection`) was being followed as a real link on Montreal's
page (now denylisted); Vector Institute's and Nymi's URLs had moved (now
corrected). Two sources are **known broken** and marked as such directly in
`sources.yaml` rather than silently dropped: Toronto's TMMIS (403 — bot-
protected) and Winnipeg's DMIS (TLS certificate failure on the city's
server). NEC's press page (403) is similarly marked broken in `sources.yaml`'s
company section.

Municipal's aggregate table row looks like "not much happening," but the
real story needed a per-source check, not just the aggregate count: it
wasn't that 15 sources were each quietly finding nothing — 9 of the original
15 weren't functioning at all (2 blocked, 7 rendering client-side and
returning no real links to a plain fetch), and those 7 have since been
dropped from `sources.yaml` rather than left in as silent dead weight — see
"Curated source research" above for the detail. The 6 that remain were
verified to actually fetch and keyword-match real content; 2 candidates
from Toronto Police Services Board is a genuine result from that working
set, not a sign the other sources are failing quietly.

## Known TODOs / open items

- Municipal coverage: Vancouver's council page and Calgary/Edmonton/Ottawa/
  Halifax's eScribe-hosted council/police-board pages (7 sources) were
  dropped from `sources.yaml` — they render client-side and return no real
  links to a plain HTTP fetch. Adding a headless-browser (Playwright) fetch
  path would let these be re-added; not done here since it's a real
  dependency/CI-complexity tradeoff, not a quick fix.
- Tag format: confirm the real convention on existing library items and fix
  `item_builder.py` if the current `practice:`/`ethics:`/`jurisdiction:`
  prefix guess doesn't match — see the note under Insert above.
- Downstream: confirm where the existing chunk/embed/index builder actually
  lives (it isn't in this repo currently) so items promoted out of Candidate
  Sources have a clear path into `out/chunks.json`.
- `scan_page` scraping is generic and per-site brittleness is expected — watch
  the first few monthly runs and add site-specific scrapers where needed.
- Failure notifications aren't wired up yet — a broken run currently only
  surfaces via GitHub's own workflow-failure email.
- `GROQ_MODEL` defaults to `llama-3.3-70b-versatile`; if a run gets
  model-not-found errors, check
  [console.groq.com/docs/models](https://console.groq.com/docs/models) for
  the current model list and override via the `GROQ_MODEL` env var.
- Groq's `response_format: json_object` guarantees valid JSON but not a
  specific schema, unlike Gemini's `responseSchema` — `judge_candidate()`
  validates the parsed result has all five expected keys and falls back to
  `_FALLBACK_VERDICT` (relevant: false) if not, but this is unverified
  against a real Groq response since there's no local key to test with.
  Worth checking the first live run's logs for
  `couldn't parse Groq response` / `missing expected keys` messages.

## Resilience: rate limits and mid-run failures

A real first live run hit exactly the failure mode you'd expect from a
free-tier LLM API and a big batch: 1,845 fresh candidates to judge, and it
crashed on a transient `503` partway through — which, at the time, threw
away *everything* that run had done (the Zotero sync, every keyword/LLM
decision made so far), guaranteeing the next run would hit the identical
wall. Fixed with three changes, all in `filter_candidates.py` /
`run_pipeline.py`, and provider-agnostic (they apply regardless of which
judge API is behind `judge_candidate()`):

- **Retry with backoff** — `429`/`500`/`502`/`503`/`504` (rate limit or
  transient overload) get retried up to 4 times with exponential backoff;
  anything else (e.g. `400`/`401`) fails immediately since retrying won't
  help.
- **Proactive throttling** — a pause between judge calls
  (`_MIN_SECONDS_BETWEEN_CALLS`, currently 2.5s for Groq's 30 RPM) keeps
  this comfortably under free-tier RPM instead of firing calls as fast as
  HTTP allows and relying on reactive 429-retries for most of them.
- **A per-run cap** (`MAX_JUDGE_CALLS_PER_RUN = 300`) — judging thousands of
  candidates in one run isn't practical anyway (long runs, throttled).
  Whatever's past the cap, or whose judge call still fails after retries, is
  left unmarked in state rather than blacklisted — picked up and judged
  again next run, so a large backlog drains over a few runs instead of
  needing to complete in one.
- **A circuit breaker** (`_CONSECUTIVE_FAILURE_CIRCUIT_BREAKER = 3`) — if
  several judge calls in a row fail even after their own retries, that's a
  strong signal the whole API is down or the key is out of quota, not that
  one candidate is unlucky. The run stops judging entirely instead of
  grinding through the rest of the cap the same way, leaving the untried
  remainder unmarked. This exists because it was needed: with Gemini as the
  original judge, one run hit exactly this — every one of 300 candidates
  429'd on all 4 attempts each, burning ~1,200 calls for zero results before
  the run gave up.
- **`run()`'s core logic is wrapped in try/finally** — `state/seen.json` and
  `state/zotero_seen.json` are saved no matter how the run exits, so a
  failure anywhere (the judge API, Zotero, network) never throws away sync
  progress or filtering decisions already made.

**Why Groq instead of Gemini**: the pattern above (every candidate in a run
uniformly hitting `429`, not just some) pointed to the API key being out of
quota for that stretch rather than a pacing issue — if it were only RPM
throttling, at least the first call of a fresh run should succeed. Gemini's
free tier couldn't sustain a 300-candidate run; Groq's (30 RPM / 1,000
requests per day for llama-3.3-70b-versatile) has much more headroom for
the same workload. `GEMINI_API_KEY` is no longer read by the pipeline.

Separately, the same live run's state-commit step failed with `Write
access to repository not granted` / `403` on `git push` —
`GITHUB_TOKEN` defaults to read-only. Fixed by adding
`permissions: contents: write` to the workflow's `watch` job. If pushes
still fail after that, check the repo's Settings → Actions → General →
"Workflow permissions" isn't locked to read-only at the repo/org level,
which a workflow-level `permissions:` block can't override.
