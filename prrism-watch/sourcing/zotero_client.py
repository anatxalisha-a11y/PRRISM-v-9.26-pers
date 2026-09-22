import time
from typing import List, Tuple

import requests


API_BASE = "https://api.zotero.org"


class ZoteroClient:
    """Thin wrapper over the Zotero Web API (v3) — plain REST, no MCP,
    since a headless GitHub Actions run has nothing to keep an MCP server
    session alive for.
    """

    def __init__(
        self,
        api_key: str,
        library_id: str,
        library_type: str = "group",
    ):
        self.api_key = api_key
        self.library_id = library_id
        self.library_type = library_type

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Zotero-API-Version": "3",
                "Authorization": f"Bearer {api_key}",
            }
        )

    def _library_path(self) -> str:
        segment = "groups" if self.library_type == "group" else "users"
        return f"{API_BASE}/{segment}/{self.library_id}"

    def _respect_backoff(self, resp: requests.Response) -> None:
        backoff = resp.headers.get("Backoff")
        if backoff:
            time.sleep(float(backoff))

        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("Retry-After", "5")))

    def _get_all_pages(self, url: str, params: dict = None) -> List[dict]:
        params = dict(params or {})
        params.setdefault("limit", 100)

        results: List[dict] = []

        next_url = url
        next_params = params

        while next_url:
            resp = self.session.get(
                next_url,
                params=next_params,
                timeout=30,
            )

            self._respect_backoff(resp)
            resp.raise_for_status()

            results.extend(resp.json())

            next_url = resp.links.get("next", {}).get("url")
            next_params = None  # already encoded into next_url

        return results

    def get_subcollections(self, collection_key: str) -> List[dict]:
        url = (
            f"{self._library_path()}"
            f"/collections/{collection_key}/collections"
        )
        return self._get_all_pages(url)

    def get_collection_items(self, collection_key: str) -> List[dict]:
        """Return all top-level items directly contained in a collection."""
        url = (
            f"{self._library_path()}"
            f"/collections/{collection_key}/items/top"
        )
        return self._get_all_pages(url)

    def get_collection_tree_keys(
        self,
        root_collection_key: str,
        _seen: set = None,
    ) -> set:
        """Return all collection keys in the root collection's subtree.

        This only walks collections, not items. It can therefore be used to
        test collection membership against an incremental item delta without
        re-fetching every item in the tree on every run.
        """
        if _seen is None:
            _seen = set()

        if root_collection_key in _seen:
            return _seen

        _seen.add(root_collection_key)

        subcollections = self.get_subcollections(root_collection_key)

        for collection in subcollections:
            collection_key = collection.get("key")

            if collection_key and collection_key not in _seen:
                self.get_collection_tree_keys(
                    collection_key,
                    _seen,
                )

        return _seen

    def get_items_since(
        self,
        since_version: int,
    ) -> Tuple[List[dict], int]:
        """Return items added or modified since a Zotero library version.

        Zotero's version-based incremental sync uses `since`. Passing 0
        naturally performs a full first-run sync.

        Returns:
            Tuple containing:
            - changed items
            - current library version
        """
        url = f"{self._library_path()}/items"

        params = {
            "since": since_version,
            "limit": 100,
        }

        results: List[dict] = []
        version = since_version

        next_url = url
        next_params = params

        while next_url:
            resp = self.session.get(
                next_url,
                params=next_params,
                timeout=30,
            )

            self._respect_backoff(resp)
            resp.raise_for_status()

            results.extend(resp.json())

            version = int(
                resp.headers.get(
                    "Last-Modified-Version",
                    version,
                )
            )

            next_url = resp.links.get("next", {}).get("url")
            next_params = None  # already encoded into next_url

        return results, version

    def get_library_tags(self) -> List[str]:
        """Return every tag currently used anywhere in the library.

        This gives the relevance judge examples of the library's existing
        tag style so newly generated tags remain consistent.
        """
        url = f"{self._library_path()}/tags"

        return [
            tag["tag"]
            for tag in self._get_all_pages(url)
            if tag.get("tag")
        ]

    def item_template(self, item_type: str) -> dict:
        resp = requests.get(
            f"{API_BASE}/items/new",
            params={"itemType": item_type},
            timeout=30,
        )

        resp.raise_for_status()
        return resp.json()

    def create_items(self, items: List[dict]) -> List[dict]:
        """POST items in batches of 50, Zotero's per-request item limit."""
        url = f"{self._library_path()}/items"

        results = []

        for i in range(0, len(items), 50):
            batch = items[i:i + 50]

            resp = self.session.post(
                url,
                json=batch,
                timeout=30,
            )

            self._respect_backoff(resp)
            resp.raise_for_status()

            results.append(resp.json())

        return results

    def download_attachment(self, item_key: str) -> bytes:
        """Download the stored file for a Zotero attachment item."""
        url = f"{self._library_path()}/items/{item_key}/file"

        resp = self.session.get(
            url,
            timeout=60,
            allow_redirects=True,
        )

        self._respect_backoff(resp)
        resp.raise_for_status()

        return resp.content