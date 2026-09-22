from datetime import date

from .config import ZOTERO_TARGET_COLLECTION
from .models import Candidate

ITEM_TYPE_BY_FAMILY = {
    "academic": "journalArticle",
    "news": "newspaperArticle",
    "government": "webpage",
    "municipal": "webpage",
    "company": "webpage",
}


def _author_to_creator(name: str) -> dict:
    parts = name.strip().split(" ", 1)
    if len(parts) == 2:
        return {"creatorType": "author", "firstName": parts[0], "lastName": parts[1]}
    return {"creatorType": "author", "name": name}


def build_zotero_item(candidate: Candidate, template: dict) -> dict:
    """Fills a Zotero item-type template (from ZoteroClient.item_template)
    with a Candidate's fields, files it into Candidate Sources, and applies
    the practice/ethics/jurisdiction tags the LLM judge assigned."""
    item = dict(template)
    item["title"] = candidate.title
    item["url"] = candidate.url
    item["accessDate"] = date.today().isoformat()
    item["collections"] = [ZOTERO_TARGET_COLLECTION]

    if candidate.published_date:
        item["date"] = candidate.published_date
    if candidate.snippet:
        item["abstractNote"] = candidate.snippet

    if candidate.source_family == "academic":
        if candidate.journal:
            item["publicationTitle"] = candidate.journal
        if candidate.doi:
            item["DOI"] = candidate.doi
        if candidate.authors:
            item["creators"] = [_author_to_creator(a) for a in candidate.authors]
    else:
        if "websiteTitle" in item:
            item["websiteTitle"] = candidate.organization or ""
        if candidate.organization and "publicationTitle" in item:
            item["publicationTitle"] = candidate.organization

    tags = [{"tag": f"source-family:{candidate.source_family}"}, {"tag": "auto-sourced"}]
    tags += [{"tag": f"practice:{t}"} for t in candidate.practice_tags]
    tags += [{"tag": f"ethics:{t}"} for t in candidate.ethics_tags]
    tags += [{"tag": f"jurisdiction:{t}"} for t in candidate.jurisdiction_tags]
    item["tags"] = tags

    return item
