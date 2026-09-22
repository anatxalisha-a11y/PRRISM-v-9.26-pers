import sys

import yaml

from .config import (
    SOURCES_YAML_PATH,
    ZOTERO_API_KEY,
    ZOTERO_DEDUP_ROOT_COLLECTION,
    ZOTERO_LIBRARY_ID,
    ZOTERO_LIBRARY_TYPE,
)
from .dedup import (
    is_duplicate,
    load_state,
    load_zotero_state,
    mark_seen,
    save_state,
    save_zotero_state,
    sync_zotero_seen_state,
    zotero_seen_keys,
)
from .discover_academic import discover as discover_academic
from .discover_curated import discover as discover_curated
from .discover_news import discover as discover_news
from .filter_candidates import filter_candidates
from .item_builder import ITEM_TYPE_BY_FAMILY, build_zotero_item
from .tags import load_taxonomy
from .zotero_client import ZoteroClient


def load_sources() -> dict:
    with open(SOURCES_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def discover_all(sources: dict) -> list:
    candidates = []
    candidates += discover_academic()
    candidates += discover_curated(sources, "government")
    candidates += discover_curated(sources, "municipal")
    candidates += discover_curated(sources, "company")
    candidates += discover_news(sources)
    return [c for c in candidates if c.url and c.title]


def run() -> None:
    if not ZOTERO_API_KEY:
        sys.exit("ZOTERO_API_KEY is not set — see prrism-watch/README.md.")

    sources = load_sources()
    taxonomy = load_taxonomy()
    state = load_state()

    zotero = ZoteroClient(ZOTERO_API_KEY, ZOTERO_LIBRARY_ID, ZOTERO_LIBRARY_TYPE)
    zotero_state = load_zotero_state()

    # state/zotero_state are saved no matter how this exits — a transient
    # failure partway through (Groq down, Zotero down, etc.) should never
    # throw away Zotero-sync progress or filtering decisions already made;
    # the next run should pick up from here, not redo all of it.
    try:
        print(f"Syncing Zotero collection tree {ZOTERO_DEDUP_ROOT_COLLECTION} (since library version {zotero_state.get('version', 0)})...")
        zotero_state = sync_zotero_seen_state(zotero, ZOTERO_DEDUP_ROOT_COLLECTION, zotero_state)
        zotero_seen = zotero_seen_keys(zotero_state)
        print(f"{len(zotero_seen)} existing items indexed for dedup (library now at version {zotero_state['version']}).")

        candidates = discover_all(sources)
        print(f"Discovered {len(candidates)} raw candidates.")

        fresh = [c for c in candidates if not is_duplicate(c, state, zotero_seen)]
        print(f"{len(fresh)} are not already in Zotero or local state.")

        existing_tags = zotero.get_library_tags()
        to_insert = filter_candidates(fresh, taxonomy, state, existing_tags)
        print(f"{len(to_insert)} candidates passed filtering and will be added to Zotero.")

        templates: dict = {}
        payload = []
        for c in to_insert:
            item_type = ITEM_TYPE_BY_FAMILY[c.source_family]
            if item_type not in templates:
                templates[item_type] = zotero.item_template(item_type)
            payload.append(build_zotero_item(c, templates[item_type]))

        if payload:
            zotero.create_items(payload)
            for c in to_insert:
                mark_seen(state, c, "inserted")

        print("Done.")
    finally:
        save_state(state)
        save_zotero_state(zotero_state)


if __name__ == "__main__":
    run()
