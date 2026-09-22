import json
from functools import lru_cache
from typing import Set

from .config import TAGS_JSON_PATH

# Not part of include/tags.json (which only defines practice/ethics/jurisdiction
# tags for *classifying* text) — this is the vocabulary used to recognize that a
# candidate page is about facial recognition at all, before it ever reaches the
# LLM judge.
FR_TERMS = [
    "facial recognition",
    "face recognition",
    "facial recognition technology",
    "biometric identification",
    "biometric surveillance",
    "face matching",
    "facial biometrics",
    "live facial recognition",
    "automated facial recognition",
]


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    with open(TAGS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def canada_terms() -> Set[str]:
    taxonomy = load_taxonomy()
    canada = taxonomy["jurisdiction_tags"]["Canada"]
    terms = set(canada.get("aliases", []))
    terms.update(canada.get("jurisdictions", []))
    return terms
