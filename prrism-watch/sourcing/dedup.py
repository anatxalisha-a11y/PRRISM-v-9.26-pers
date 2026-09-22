import json
import os
from datetime import date
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .config import STATE_PATH, ZOTERO_STATE_PATH
from .models import Candidate

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "mc_cid", "mc_eid",
}


def normalize_url(url: str) -> str:
    """Strip scheme case, www., trailing slash, and tracking params so the
    same article reached two different ways still dedups to one key."""
    if not url:
        return ""
    parts = urlsplit(url.strip())
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    query = sorted(
        (k, v) for k, v in parse_qsl(parts.query) if k.lower() not in TRACKING_PARAMS
    )
    return urlunsplit(("https", netloc, path, urlencode(query), ""))


def dedup_key(candidate: Candidate) -> str:
    if candidate.doi:
        return f"doi:{candidate.doi.strip().lower()}"
    return f"url:{normalize_url(candidate.url)}"


def load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def mark_seen(state: dict, candidate: Candidate, status: str) -> None:
    state[dedup_key(candidate)] = {
        "title": candidate.title,
        "url": candidate.url,
        "doi": candidate.doi,
        "source_family": candidate.source_family,
        "status": status,  # "inserted" | "rejected_keyword" | "rejected_llm"
        "checked_at": date.today().isoformat(),
    }


def load_zotero_state() -> dict:
    if not os.path.exists(ZOTERO_STATE_PATH):
        return {"version": 0, "items": {}}
    with open(ZOTERO_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_zotero_state(zotero_state: dict) -> None:
    os.makedirs(os.path.dirname(ZOTERO_STATE_PATH), exist_ok=True)
    with open(ZOTERO_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(zotero_state, f, indent=2, sort_keys=True)
        f.write("\n")


def sync_zotero_seen_state(zotero_client, root_collection_key: str, zotero_state: dict) -> dict:
    """Updates zotero_state in place-ish (returns it) instead of walking every
    item in the collection tree on every run. Only the *first* run (no
    persisted version, so since=0) fetches everything; every run after asks
    Zotero for just what changed since the last recorded library version —
    Zotero's version-based incremental sync is designed for exactly this.
    Collections themselves (a handful, not items) are still walked fresh
    each run to know which keys count as "in the tree," since that's cheap.

    Known limitation: this tracks *membership*, not permanent deletion — an
    item removed from the tree (moved to a collection outside it) drops out
    on its next version bump, but an item permanently deleted from Zotero
    entirely leaves a stale entry behind (its URL stays treated as seen).
    Low-risk default (favors not re-adding over accidentally losing a
    dedup); if that ever bites, delete the stale entry from
    state/zotero_seen.json by hand.
    """
    tree_keys = zotero_client.get_collection_tree_keys(root_collection_key)
    since = zotero_state.get("version", 0)
    items = zotero_state.setdefault("items", {})

    changed, new_version = zotero_client.get_items_since(since)
    for item in changed:
        data = item.get("data", {})
        key = data.get("key")
        if not key:
            continue
        if set(data.get("collections", [])) & tree_keys:
            items[key] = {"url": data.get("url"), "doi": data.get("DOI")}
        else:
            items.pop(key, None)  # no longer in the tree (or never was)

    zotero_state["version"] = new_version
    return zotero_state


def zotero_seen_keys(zotero_state: dict) -> set:
    keys = set()
    for entry in zotero_state.get("items", {}).values():
        doi = entry.get("doi")
        url = entry.get("url")
        if doi:
            keys.add(f"doi:{doi.strip().lower()}")
        if url:
            keys.add(f"url:{normalize_url(url)}")
    return keys


def is_duplicate(candidate: Candidate, local_state: dict, zotero_seen_set: set) -> bool:
    key = dedup_key(candidate)
    return key in local_state or key in zotero_seen_set
