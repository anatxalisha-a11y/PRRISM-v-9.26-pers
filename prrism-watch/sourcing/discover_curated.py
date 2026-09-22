from io import BytesIO
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

import feedparser
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import USER_AGENT
from .models import Candidate
from .tags import FR_TERMS

# Shared discovery for the curated source families (government, municipal,
# company). Each entry in sources.yaml is one of:
#
#   - {name, rss: <feed url>}
#       Parsed directly, every entry is a candidate.
#
#   - {name, scan_page: <page url>}
#       ONE-HOP: fetched once, links whose *anchor text* mentions facial
#       recognition/biometrics become candidates. Fits content-listing pages
#       (blogs, newsrooms, press pages) where the link text already is the
#       article title.
#
#   - {name, scan_index: <list url>, detail_link_pattern?, max_detail_pages?}
#       TWO-HOP: fits government/municipal meeting-agenda listings, where the
#       index page's links are just dates ("March 4, 2025") and the actual
#       agenda item titles ("Facial Recognition System Audit") only appear on
#       each meeting's own page. Follows same-domain links from the index
#       (optionally narrowed by detail_link_pattern, a required substring),
#       caps how many it fetches per run (max_detail_pages, default 10), and
#       keyword-matches the *full text* of each detail page rather than just
#       link text.
#
# A source can combine these (e.g. rss as primary, scan_page as fallback).

_NAV_PATH_DENYLIST = (
    "privacy", "accessibility", "contact", "about", "careers", "sitemap",
    "terms", "cookie", "login", "search", "subscribe", "français", "francais",
    "cdn-cgi",  # Cloudflare's obfuscated-email links — never real content
)


def _absolutize(href: str, base: str) -> Optional[str]:
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return base + href
    if href.startswith("http"):
        return href
    return None


def _fetch(url: str) -> requests.Response:
    resp = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp


def _extract_text_and_title(resp: requests.Response) -> Tuple[str, str]:
    """Government/municipal agendas and minutes are routinely published as
    PDFs rather than HTML pages (e.g. TPSB's wp-content/uploads/*.pdf) — a
    plain BeautifulSoup parse gets nothing from those, so PDFs get their own
    text-extraction path."""
    content_type = resp.headers.get("Content-Type", "")
    if resp.url.lower().endswith(".pdf") or "application/pdf" in content_type:
        title = resp.url.rsplit("/", 1)[-1]
        try:
            reader = PdfReader(BytesIO(resp.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:  # noqa: BLE001 - a malformed PDF shouldn't kill the run
            print(f"[discover_curated] PDF text extraction failed for {resp.url}: {e}")
            text = ""
        return text, title

    soup = BeautifulSoup(resp.text, "html.parser")
    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else resp.url
    return soup.get_text(" ", strip=True), title


def _from_rss(source: dict, family: str) -> List[Candidate]:
    feed = feedparser.parse(source["rss"])
    out = []
    for entry in feed.entries:
        out.append(Candidate(
            title=entry.get("title", ""),
            url=entry.get("link", ""),
            source_family=family,
            organization=source.get("name"),
            published_date=entry.get("published", entry.get("updated")),
            snippet=entry.get("summary"),
        ))
    return out


def _from_page_scan(source: dict, family: str) -> List[Candidate]:
    resp = _fetch(source["scan_page"])
    soup = BeautifulSoup(resp.text, "html.parser")
    parsed = urlsplit(source["scan_page"])
    base = f"{parsed.scheme}://{parsed.netloc}"

    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if not text:
            continue
        lowered = text.lower()
        if not any(term.lower() in lowered for term in FR_TERMS) and "biometric" not in lowered:
            continue
        href = _absolutize(a["href"], base)
        if not href or href in seen:
            continue
        seen.add(href)
        out.append(Candidate(
            title=text, url=href, source_family=family, organization=source.get("name"),
        ))
    return out


def _from_index_scan(source: dict, family: str) -> List[Candidate]:
    index_url = source["scan_index"]
    pattern = source.get("detail_link_pattern")
    max_pages = source.get("max_detail_pages", 10)

    resp = _fetch(index_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    parsed = urlsplit(index_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    candidate_links, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = _absolutize(a["href"], base)
        if not href or urlsplit(href).netloc != parsed.netloc:
            continue
        if href in seen or href.rstrip("/") == index_url.rstrip("/"):
            continue
        if pattern and pattern not in href:
            continue
        if not pattern and any(bad in href.lower() for bad in _NAV_PATH_DENYLIST):
            continue
        seen.add(href)
        candidate_links.append(href)

    if not pattern:
        # No site-specific pattern to trust — meeting/agenda links almost
        # always carry a date or numeric id, plain nav links usually don't.
        candidate_links.sort(key=lambda h: not any(ch.isdigit() for ch in h))
    detail_links = candidate_links[:max_pages]

    out = []
    for link in detail_links:
        try:
            detail_resp = _fetch(link)
        except Exception as e:  # noqa: BLE001 - one broken detail page shouldn't kill the run
            print(f"[discover_curated] detail fetch failed for {link}: {e}")
            continue
        page_text, title = _extract_text_and_title(detail_resp)
        lowered = page_text.lower()
        if not any(term.lower() in lowered for term in FR_TERMS) and "biometric" not in lowered:
            continue
        out.append(Candidate(
            title=title, url=link, source_family=family, organization=source.get("name"),
        ))
    return out


def discover_from_source(source: dict, family: str) -> List[Candidate]:
    candidates = []
    if source.get("rss"):
        try:
            candidates.extend(_from_rss(source, family))
        except Exception as e:  # noqa: BLE001
            print(f"[discover_curated] RSS failed for {source.get('name')}: {e}")
    if source.get("scan_page"):
        try:
            candidates.extend(_from_page_scan(source, family))
        except Exception as e:  # noqa: BLE001
            print(f"[discover_curated] page scan failed for {source.get('name')}: {e}")
    if source.get("scan_index"):
        try:
            candidates.extend(_from_index_scan(source, family))
        except Exception as e:  # noqa: BLE001
            print(f"[discover_curated] index scan failed for {source.get('name')}: {e}")
    return candidates


def discover(sources_yaml: dict, family_key: str) -> List[Candidate]:
    candidates = []
    for source in sources_yaml.get(family_key, []):
        candidates.extend(discover_from_source(source, source.get("source_family", family_key)))
    return candidates
