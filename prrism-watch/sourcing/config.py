import os

SOURCING_DIR = os.path.dirname(os.path.abspath(__file__))
WATCH_ROOT = os.path.dirname(SOURCING_DIR)
REPO_ROOT = os.path.dirname(WATCH_ROOT)

STATE_PATH = os.path.join(WATCH_ROOT, "state", "seen.json")
# Mirrors which Zotero items (by key) are currently inside the dedup
# collection tree, plus the library version last synced to — lets dedup.py
# ask Zotero for only what changed since last run instead of re-walking the
# whole tree's items every time.
ZOTERO_STATE_PATH = os.path.join(WATCH_ROOT, "state", "zotero_seen.json")
SOURCES_YAML_PATH = os.path.join(SOURCING_DIR, "sources.yaml")
TAGS_JSON_PATH = os.path.join(REPO_ROOT, "include", "tags.json")

# Groq's free tier (30 RPM / 1,000 requests per day on this model) is what
# the relevance judge runs on — replaced Gemini after real runs showed its
# free tier couldn't sustain a 300-candidate batch (see README.md
# "Resilience"). llama-3.3-70b-versatile was the original default here but
# Groq deprecated it (June 2026) in favor of this one — same free-tier
# limits, per Groq's docs. If GROQ_MODEL ever 404s with
# "model_decommissioned" again, check console.groq.com/docs/models for the
# current recommendation.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# uai_lab group library — https://www.zotero.org/groups/5614036/uai_lab
ZOTERO_API_KEY = os.environ.get("ZOTERO_API_KEY")
ZOTERO_LIBRARY_ID = os.environ.get("ZOTERO_LIBRARY_ID", "5614036")
ZOTERO_LIBRARY_TYPE = os.environ.get("ZOTERO_LIBRARY_TYPE", "group")  # "group" or "user"

# "Candidate Sources" collection — new items land here for manual triage.
ZOTERO_TARGET_COLLECTION = os.environ.get("ZOTERO_TARGET_COLLECTION", "CSZS3XH5")
# Parent collection whose subtree (Candidate Sources, Benchmarks, Best Practices,
# Policy) is checked for dedup, so items already promoted out of Candidate Sources
# don't get re-added on a later run.
ZOTERO_DEDUP_ROOT_COLLECTION = os.environ.get("ZOTERO_DEDUP_ROOT_COLLECTION", "HBREESE8")

# News discovery search API — see README.md "Tavily credit budget" for cost.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Optional, opt-in only — no default. If set, this is sent to third parties:
# as the OpenAlex API's "mailto" polite-pool param, and in the User-Agent
# header on every scrape request to government/municipal/company sites. Set
# it to a project/team contact if you want one identified in that traffic —
# never defaults to a personal email.
CONTACT_EMAIL = os.environ.get("WATCH_CONTACT_EMAIL")
USER_AGENT = (
    f"prrism-watch/0.1 (Canadian facial recognition primary source bot; contact {CONTACT_EMAIL})"
    if CONTACT_EMAIL
    else "prrism-watch/0.1 (Canadian facial recognition primary source bot)"
)
