import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# prrism-build/ and prrism-watch/ are sibling directories.
# Add prrism-watch so we can reuse the existing Zotero client.
REPO_ROOT = Path(__file__).resolve().parents[1]
WATCH_DIR = REPO_ROOT / "prrism-watch"

if str(WATCH_DIR) not in sys.path:
    sys.path.insert(0, str(WATCH_DIR))

from sourcing.zotero_client import ZoteroClient


ZOTERO_LIBRARY_ID = os.environ.get(
    "ZOTERO_LIBRARY_ID",
    "5614036",
)
ZOTERO_LIBRARY_TYPE = os.environ.get(
    "ZOTERO_LIBRARY_TYPE",
    "group",
)
ZOTERO_ROOT_COLLECTION = os.environ.get(
    "ZOTERO_ROOT_COLLECTION",
    "HBREESE8",
)

APPROVED_COLLECTION_NAMES = {
    "Benchmarks",
    "Best Practices",
    "Policy",
}


@dataclass(frozen=True)
class Source:
    source_id: str
    url: str
    title: str
    item_type: str
    collections: tuple[str, ...]
    content_type: str = ""
    link_mode: str = ""
    attachment_key: str = ""


def resolve_approved_collections(
    zotero: ZoteroClient,
    root_collection_key: str = ZOTERO_ROOT_COLLECTION,
) -> Dict[str, str]:
    """
    Resolve the three approved PRRISM collection roots.

    Returns:
        {
            "Benchmarks": "<key>",
            "Best Practices": "<key>",
            "Policy": "<key>",
        }

    Candidate Sources is intentionally not included.
    """
    subcollections = zotero.get_subcollections(
        root_collection_key
    )

    resolved: Dict[str, str] = {}

    for collection in subcollections:
        data = collection.get("data", {})

        name = data.get("name")
        key = data.get("key") or collection.get("key")

        if name in APPROVED_COLLECTION_NAMES and key:
            resolved[name] = key

    missing = APPROVED_COLLECTION_NAMES - set(resolved)

    if missing:
        missing_names = ", ".join(sorted(missing))

        raise RuntimeError(
            "Could not find all approved Zotero collections under "
            f"{root_collection_key}. Missing: {missing_names}"
        )

    return resolved


def walk_collection_tree(
    zotero: ZoteroClient,
    collection_key: str,
    collection_name: str,
):
    """
    Yield one collection and every descendant collection recursively.

    Each yielded value is:

        (collection_key, collection_name)

    This means an approved root such as Best Practices includes both
    items directly inside Best Practices and items inside descendants
    such as Amazon, Apple, Microsoft, etc.
    """
    yield collection_key, collection_name

    children = zotero.get_subcollections(collection_key)

    for child in children:
        data = child.get("data", {})

        child_key = data.get("key") or child.get("key")
        child_name = (data.get("name") or "").strip()

        if not child_key:
            continue

        if not child_name:
            child_name = child_key

        yield from walk_collection_tree(
            zotero,
            collection_key=child_key,
            collection_name=child_name,
        )


def _item_to_source(
    item: dict,
    collection_name: str,
) -> Source | None:
    """
    Convert a Zotero API item into a PRRISM Source.

    Sources may be:
      - normal web/linked sources with a URL
      - files stored directly in Zotero

    Stored Zotero files do not need an external URL because the builder
    can retrieve them through Zotero using their attachment key.
    """
    data = item.get("data", {})

    source_id = item.get("key") or data.get("key")
    url = (data.get("url") or "").strip()
    title = (data.get("title") or "").strip()
    item_type = (data.get("itemType") or "").strip()
    content_type = (data.get("contentType") or "").strip()
    link_mode = (data.get("linkMode") or "").strip()

    is_stored_file = (
        item_type == "attachment"
        and link_mode == "imported_file"
    )

    # A source must either have an external URL or be a file
    # stored directly in Zotero.
    if not source_id or (not url and not is_stored_file):
        return None

    if not title:
        title = (
            (data.get("filename") or "").strip()
            or url
            or source_id
        )

    return Source(
        source_id=source_id,
        url=url,
        title=title,
        item_type=item_type,
        collections=(collection_name,),
        content_type=content_type,
        link_mode=link_mode,
        attachment_key=(
            source_id if is_stored_file else ""
        ),
    )


def load_approved_sources(
    zotero: ZoteroClient,
    root_collection_key: str = ZOTERO_ROOT_COLLECTION,
) -> List[Source]:
    """
    Load the current human-approved PRRISM corpus from Zotero.

    Approval is determined by membership anywhere within the
    collection trees rooted at:

      - Benchmarks
      - Best Practices
      - Policy

    All descendant collections are included recursively.

    Candidate Sources is intentionally ignored because it is not
    underneath any approved collection root.

    Zotero items appearing in more than one approved collection are
    deduplicated by Zotero item key while preserving all collection
    memberships.
    """
    approved_roots = resolve_approved_collections(
        zotero,
        root_collection_key=root_collection_key,
    )

    # Zotero item key -> mutable representation while collection
    # memberships are merged.
    merged: Dict[str, dict] = {}

    for approved_root_name in sorted(approved_roots):
        approved_root_key = approved_roots[
            approved_root_name
        ]

        for collection_key, collection_name in walk_collection_tree(
            zotero,
            collection_key=approved_root_key,
            collection_name=approved_root_name,
        ):
            items = zotero.get_collection_items(
                collection_key
            )

            for item in items:
                source = _item_to_source(
                    item,
                    collection_name,
                )

                if source is None:
                    continue

                if source.source_id not in merged:
                    merged[source.source_id] = {
                        "source_id": source.source_id,
                        "url": source.url,
                        "title": source.title,
                        "item_type": source.item_type,
                        "content_type": source.content_type,
                        "link_mode": source.link_mode,
                        "attachment_key": (
                            source.attachment_key
                        ),
                        "collections": set(
                            source.collections
                        ),
                    }

                else:
                    merged[
                        source.source_id
                    ]["collections"].update(
                        source.collections
                    )

    sources = [
        Source(
            source_id=record["source_id"],
            url=record["url"],
            title=record["title"],
            item_type=record["item_type"],
            collections=tuple(
                sorted(record["collections"])
            ),
            content_type=record["content_type"],
            link_mode=record["link_mode"],
            attachment_key=record[
                "attachment_key"
            ],
        )
        for record in merged.values()
    ]

    # Deterministic output regardless of Zotero API ordering.
    sources.sort(
        key=lambda source: source.source_id
    )

    if not sources:
        raise RuntimeError(
            "No approved PRRISM sources were found in Zotero. "
            "Refusing to continue with an empty corpus."
        )

    return sources


def main() -> None:
    api_key = os.environ.get("ZOTERO_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ZOTERO_API_KEY is not set."
        )

    zotero = ZoteroClient(
        api_key=api_key,
        library_id=ZOTERO_LIBRARY_ID,
        library_type=ZOTERO_LIBRARY_TYPE,
    )

    print(
        "Loading approved PRRISM corpus "
        "from Zotero...\n"
    )

    approved_roots = resolve_approved_collections(
        zotero
    )

    print("Approved collection roots:")

    for name in sorted(approved_roots):
        print(
            f"  {name:<16} "
            f"{approved_roots[name]}"
        )

    print("\nResolved approved collection trees:")

    for root_name in sorted(approved_roots):
        root_key = approved_roots[root_name]

        for collection_key, collection_name in (
            walk_collection_tree(
                zotero,
                collection_key=root_key,
                collection_name=root_name,
            )
        ):
            prefix = (
                "  ROOT"
                if collection_key == root_key
                else "    └─"
            )

            print(
                f"{prefix} "
                f"{collection_name} "
                f"[{collection_key}]"
            )

    sources = load_approved_sources(zotero)

    print(
        f"\nUnique approved sources: "
        f"{len(sources)}\n"
    )

    for source in sources:
        memberships = ", ".join(
            source.collections
        )

        source_kind = (
            "stored Zotero file"
            if source.attachment_key
            else "URL"
        )

        print(
            f"[{source.source_id}] "
            f"{source.title} "
            f"({memberships}) "
            f"[{source_kind}]"
        )

    print("\nCorpus ready.")


if __name__ == "__main__":
    main()