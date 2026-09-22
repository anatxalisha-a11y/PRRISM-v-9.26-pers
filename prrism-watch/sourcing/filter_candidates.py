import json
import time
from typing import List

import requests

from .config import GROQ_API_KEY, GROQ_MODEL
from .models import Candidate
from .tags import FR_TERMS, canada_terms

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# 429/500/502/503/504 are all worth a retry (rate limit or transient
# overload); anything else (400 bad request, 401/403 bad key) won't fix
# itself.
_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4
_BASE_BACKOFF_SECONDS = 2

# Groq's free tier for llama-3.3-70b-versatile is 30 requests/minute — 2.5s
# keeps this comfortably under that (~24 RPM) instead of firing calls as
# fast as HTTP allows and relying on reactive 429-retries for most of them.
_MIN_SECONDS_BETWEEN_CALLS = 2.5

# A large first run (or a backlog after any gap) can easily produce
# thousands of keyword-pre-filtered candidates — judging all of them in one
# run, sequentially, throttled, isn't practical (long runs) and risks
# running into the exact kind of transient failure that crashed an earlier
# run mid-batch. Cap it; whatever's left past the cap is simply left
# unmarked in state, so it's picked up and judged in a future run instead —
# the backlog drains over a few runs rather than needing to complete in one.
MAX_JUDGE_CALLS_PER_RUN = 300

# If several candidates in a row fail even after _post_to_groq's own
# retries, that's not "this one candidate is unlucky" — it's a strong
# signal the API is down or the key is out of quota for the day (this
# exact thing happened on Gemini, the original provider here: every one of
# 300 candidates 429'd on all 4 attempts each, burning ~1,200 calls before
# giving up with nothing to show for it). Stop early instead of grinding
# through the rest of the cap the same way — whatever's left just stays
# unmarked for next run, same as the cap above.
_CONSECUTIVE_FAILURE_CIRCUIT_BREAKER = 3


def keyword_prefilter_pass(candidate: Candidate) -> bool:
    """Cheap first pass before any LLM call. Government/municipal/company
    candidates come from a curated, already-Canada-scoped source list, so
    only the facial-recognition term is required there; news/academic cast
    a wider net and need both an FR term and a Canada term."""
    text = " ".join(filter(None, [candidate.title, candidate.snippet])).lower()
    has_fr = any(term.lower() in text for term in FR_TERMS)
    if not has_fr:
        return False
    if candidate.source_family in ("government", "municipal", "company"):
        return True
    return any(term.lower() in text for term in canada_terms())


# Groq's OpenAI-compatible API only guarantees valid-JSON output via
# response_format: json_object (not a schema-enforced shape the way
# Gemini's responseSchema was) — so the exact required shape is spelled out
# in the prompt itself instead, and the parsed result is validated/defaulted
# defensively below rather than trusted to match.
_REQUIRED_VERDICT_KEYS = ("relevant", "reason", "practice_tags", "ethics_tags", "jurisdiction_tags")


def _build_prompt(candidate: Candidate, taxonomy: dict, existing_tags: List[str] = None) -> str:
    existing_tags_block = ""
    if existing_tags:
        existing_tags_block = f"""

For reference, tags already in use elsewhere in this Zotero library (a sample):
{existing_tags[:200]}
The practice_tags/ethics_tags/jurisdiction_tags values you assign below get \
written with a "practice:"/"ethics:"/"jurisdiction:" prefix — this is a \
placeholder convention, not confirmed against how the library actually \
tags things yet."""

    return f"""Candidate source:
Title: {candidate.title}
URL: {candidate.url}
Organization: {candidate.organization or "unknown"}
Source family: {candidate.source_family}
Snippet/abstract: {candidate.snippet or "(none)"}

Available tags (only use tags from these lists, lowercase, exactly as given):
practice_tags: {list(taxonomy["practice_tags"].keys())}
ethics_tags: {list(taxonomy["ethics_tags"].keys())}
jurisdiction_tags: {list(taxonomy["jurisdiction_tags"].keys())}
{existing_tags_block}

Decide whether this is a genuine primary source on facial recognition in \
Canada — an official regulation, proposed law, government guidance/\
investigation, an academic study, or a company's own stated policy/\
position — as opposed to secondary commentary, a passing mention, an \
unrelated product page, or content about another country. Assign whichever \
tags from the lists above actually apply; empty tag lists are fine if none \
apply.

Respond with ONLY a single JSON object, no other text, with exactly these \
keys:
- "relevant": boolean
- "reason": string, one sentence
- "practice_tags": array of strings
- "ethics_tags": array of strings
- "jurisdiction_tags": array of strings"""


_FALLBACK_VERDICT = {
    "relevant": False,
    "reason": "model did not return a usable verdict",
    "practice_tags": [],
    "ethics_tags": [],
    "jurisdiction_tags": [],
}


def _post_to_groq(payload: dict) -> requests.Response:
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()  # non-2xx -> HTTPError, caught below
            return resp
        except requests.exceptions.RequestException as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status not in _RETRIABLE_STATUS_CODES:
                raise  # e.g. 400/401 — retrying won't help
            if attempt < _MAX_RETRIES - 1:
                backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(f"[filter_candidates] Groq call failed ({e}), retrying in {backoff}s "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(backoff)
    raise last_exc


def judge_candidate(candidate: Candidate, taxonomy: dict, existing_tags: List[str] = None) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set — see prrism-watch/README.md.")
    resp = _post_to_groq({
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON API. Respond with only a single JSON object, no other text.",
            },
            {"role": "user", "content": _build_prompt(candidate, taxonomy, existing_tags)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    })
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
        verdict = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"[filter_candidates] couldn't parse Groq response for {candidate.url}: {e}")
        return dict(_FALLBACK_VERDICT)

    if not all(key in verdict for key in _REQUIRED_VERDICT_KEYS):
        print(f"[filter_candidates] Groq response missing expected keys for {candidate.url}: {verdict!r}")
        return dict(_FALLBACK_VERDICT)
    return verdict


def filter_candidates(
    candidates: List[Candidate], taxonomy: dict, state: dict, existing_tags: List[str] = None
) -> List[Candidate]:
    """Runs the two-stage filter over candidates not already in local dedup
    state. Rejections (keyword or LLM) are recorded in `state` so they aren't
    re-judged next run. Returns the candidates that passed, tagged.

    The keyword pass runs over every candidate (cheap); the Groq judge pass
    is capped at MAX_JUDGE_CALLS_PER_RUN and throttled between calls —
    anything past the cap, or whose judge call fails even after retries, is
    left unmarked in state so it's picked up again next run rather than
    either crashing the whole batch or silently writing it off."""
    from .dedup import mark_seen

    to_judge = []
    for c in candidates:
        if not keyword_prefilter_pass(c):
            mark_seen(state, c, "rejected_keyword")
            continue
        to_judge.append(c)

    if len(to_judge) > MAX_JUDGE_CALLS_PER_RUN:
        deferred = len(to_judge) - MAX_JUDGE_CALLS_PER_RUN
        print(f"[filter_candidates] {len(to_judge)} candidates passed the keyword pre-filter; "
              f"judging this run's cap of {MAX_JUDGE_CALLS_PER_RUN}, leaving {deferred} for next run")
        to_judge = to_judge[:MAX_JUDGE_CALLS_PER_RUN]

    passed = []
    consecutive_failures = 0
    for i, c in enumerate(to_judge):
        is_last = i == len(to_judge) - 1
        try:
            verdict = judge_candidate(c, taxonomy, existing_tags)
        except Exception as e:  # noqa: BLE001 - one candidate's failure shouldn't sink the whole run
            consecutive_failures += 1
            print(f"[filter_candidates] judge failed for {c.url}, leaving for next run: {e}")
            if consecutive_failures >= _CONSECUTIVE_FAILURE_CIRCUIT_BREAKER:
                remaining = len(to_judge) - i - 1
                print(f"[filter_candidates] {consecutive_failures} judge calls in a row failed — "
                      f"the judge API looks down or out of quota for now, stopping early instead "
                      f"of grinding through the rest; {remaining} candidates left for next run")
                break
            if not is_last:
                time.sleep(_MIN_SECONDS_BETWEEN_CALLS)
            continue

        consecutive_failures = 0
        if not is_last:
            time.sleep(_MIN_SECONDS_BETWEEN_CALLS)

        if not verdict.get("relevant"):
            mark_seen(state, c, "rejected_llm")
            continue
        c.relevant = True
        c.reason = verdict.get("reason")
        c.practice_tags = verdict.get("practice_tags", [])
        c.ethics_tags = verdict.get("ethics_tags", [])
        c.jurisdiction_tags = verdict.get("jurisdiction_tags", [])
        passed.append(c)
    return passed
