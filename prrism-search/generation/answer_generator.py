"""Grounded ranked-result generation for PRRISM.

This module prepares ranked PRRISM retrieval results for
researcher-facing presentation.

Retrieved chunks are grouped by their underlying Zotero source.
The ranking established by PRRISM's retrieval, reranking, and
diversity pipeline is preserved.

The language model does not determine source rank, source identity,
title, or URL. Its role is limited to explaining what each ranked
source contributes to the researcher's question.

PRRISM taxonomy labels are preserved as analytical context, while
internal similarity scores and retrieval metadata remain hidden from
the generation model and public interface.
"""

import json
import os
from typing import Dict, List

from groq import Groq


ANSWER_MODEL = os.getenv(
    "PRRISM_ANSWER_MODEL",
    "openai/gpt-oss-120b",
)


# =============================================================================
# EVIDENCE PREPARATION
# =============================================================================

def _tag_names(
    tags: List[Dict],
) -> List[str]:
    """Return taxonomy labels without exposing similarity scores."""

    return [
        tag["tag"]
        for tag in (tags or [])
        if tag.get("tag")
    ]


def _add_unique(
    target: List[str],
    values: List[str],
) -> None:
    """Append values while preserving first-seen order."""

    for value in values:
        if value not in target:
            target.append(value)


def build_evidence(
    results: List[Dict],
) -> List[Dict]:
    """Convert ranked retrieval results into ranked document evidence.

    Each unique source receives one citation ID:

        S1
        S2
        S3
        ...

    The IDs follow the order established by PRRISM's retrieval,
    reranking, and diversity pipeline.

    Multiple retrieved chunks from the same source are grouped as
    passages beneath that source.

    Internal retrieval scores and ranks are deliberately excluded from
    the generation payload. Their ordering is preserved structurally
    instead.
    """

    sources = []
    source_lookup = {}

    for result in results:

        source_id = result.get(
            "source_id"
        )

        if not source_id:
            raise ValueError(
                "Retrieved result is missing source_id."
            )

        if source_id not in source_lookup:

            source = {
                "citation_id": (
                    f"S{len(sources) + 1}"
                ),
                "source_id": source_id,
                "title": (
                    result.get("title")
                    or "Untitled source"
                ),
                "url": result.get("url"),
                "collections": list(
                    result.get("collections")
                    or []
                ),
                "themes": {
                    "ethics": [],
                    "practice": [],
                    "jurisdiction": [],
                },
                "passages": [],
            }

            sources.append(
                source
            )

            source_lookup[
                source_id
            ] = source

        source = source_lookup[
            source_id
        ]

        text = result.get(
            "text"
        )

        if text:
            source["passages"].append(
                text
            )

        _add_unique(
            source["themes"]["ethics"],
            _tag_names(
                result.get(
                    "ethics_tags",
                    [],
                )
            ),
        )

        _add_unique(
            source["themes"]["practice"],
            _tag_names(
                result.get(
                    "practice_tags",
                    [],
                )
            ),
        )

        _add_unique(
            source["themes"]["jurisdiction"],
            _tag_names(
                result.get(
                    "jurisdiction_tags",
                    [],
                )
            ),
        )

    return sources


# =============================================================================
# GENERATION PROMPT
# =============================================================================

def build_generation_prompt(
    question: str,
    evidence: List[Dict],
) -> str:
    """Build the prompt for ranked source explanations."""

    if not question or not question.strip():
        raise ValueError(
            "Question must not be empty."
        )

    if not evidence:
        raise ValueError(
            "Evidence must not be empty."
        )

    evidence_blocks = []

    for source in evidence:

        lines = [
            f"[{source['citation_id']}]",
            f"Title: {source['title']}",
        ]

        collections = (
            source.get("collections")
            or []
        )

        if collections:
            lines.append(
                "Collection: "
                + ", ".join(
                    collections
                )
            )

        themes = source.get(
            "themes",
            {},
        )

        theme_parts = []

        for category in (
            "ethics",
            "practice",
            "jurisdiction",
        ):

            values = (
                themes.get(category)
                or []
            )

            if values:
                theme_parts.extend(
                    values
                )

        if theme_parts:
            lines.append(
                "PRRISM themes: "
                + ", ".join(
                    dict.fromkeys(
                        theme_parts
                    )
                )
            )

        for number, passage in enumerate(
            source.get(
                "passages",
                [],
            ),
            start=1,
        ):

            lines.extend(
                [
                    "",
                    f"Passage {number}:",
                    passage,
                ]
            )

        evidence_blocks.append(
            "\n".join(
                lines
            )
        )

    evidence_text = (
        "\n\n---\n\n".join(
            evidence_blocks
        )
    )

    return f"""Explain what each ranked PRRISM source contributes to the researcher's question.

The sources are already ranked by PRRISM. Preserve their IDs and order exactly. Do not rerank, combine, omit, or add sources.

For each source, write a clear, natural, research-oriented explanation of what that source contributes to the question.

Use the amount of detail and structure that best reflects the supplied evidence. A focused source may require only one paragraph, while a source that contributes several distinct findings, arguments, concerns, recommendations, or themes may require multiple paragraphs.

Prioritize the source's main themes, findings, arguments, concerns, and recommendations that are relevant to the researcher's question. Develop these points sufficiently for a researcher to understand not only that the source is relevant, but what it actually contributes to the topic.

Include useful context from the supplied passages when it helps explain the source's contribution. Where the evidence supports several distinct relevant ideas, explain those ideas rather than collapsing them into a single summary sentence.

Do not force every source into the same length or structure. Do not pad a source when the retrieved evidence is limited, and do not compress substantial evidence merely to keep the response short.

Each explanation must use only passages belonging to that source. Do not transfer claims, facts, interpretations, or recommendations from one source to another.

Preserve the source's framing, uncertainty, and level of certainty. If a source makes an argument, recommendation, interpretation, or contested claim, attribute it to that source rather than presenting it as PRRISM's own conclusion.

PRRISM themes are analytical context, not evidence. Use them to help identify potentially important dimensions of the source, but mention a theme in the explanation only when the supplied passages substantively support it. Do not simply list taxonomy labels.

Do not use outside knowledge. Do not mention retrieval scores, reranking, embeddings, taxonomy machinery, or other internal PRRISM mechanics.

Do not create titles, headings, source names, URLs, citation numbers, or rankings. PRRISM will attach trusted source metadata and rank information separately.

If a source's passages do not contain enough information to explain its relevance to the question, say that the retrieved passages provide limited evidence rather than filling the gap.

Return JSON with exactly one field:

{{
  "results": [
    {{
      "citation_id": "S1",
      "summary": "natural research-oriented explanation of what this source contributes"
    }},
    {{
      "citation_id": "S2",
      "summary": "natural research-oriented explanation of what this source contributes"
    }}
  ]
}}

Return exactly one result for every supplied source, in exactly the same order.

QUESTION:
{question.strip()}

RANKED EVIDENCE:

{evidence_text}
"""


# =============================================================================
# VALIDATION
# =============================================================================

def validate_generated_answer(
    result: Dict,
    evidence: List[Dict],
) -> Dict:
    """Validate generated explanations against ranked PRRISM evidence.

    The model is allowed to generate only:

        citation_id
        summary

    Rank and trusted source metadata are attached by PRRISM after
    validation.
    """

    if set(result.keys()) != {
        "results"
    }:
        raise ValueError(
            "Generated answer has invalid fields."
        )

    generated_results = result[
        "results"
    ]

    if not isinstance(
        generated_results,
        list,
    ):
        raise ValueError(
            "Generated results must be a list."
        )

    expected_ids = [
        source["citation_id"]
        for source in evidence
    ]

    if len(generated_results) != len(
        expected_ids
    ):
        raise ValueError(
            "Generated result count does not match "
            "the ranked evidence."
        )

    generated_ids = []

    for generated in generated_results:

        if not isinstance(
            generated,
            dict,
        ):
            raise ValueError(
                "Each generated result must be an object."
            )

        if set(
            generated.keys()
        ) != {
            "citation_id",
            "summary",
        }:
            raise ValueError(
                "Generated ranked result has "
                "invalid fields."
            )

        citation_id = generated[
            "citation_id"
        ]

        summary = generated[
            "summary"
        ]

        if not isinstance(
            citation_id,
            str,
        ):
            raise ValueError(
                "citation_id must be a string."
            )

        if not isinstance(
            summary,
            str,
        ):
            raise ValueError(
                "Summary must be a string."
            )

        if not summary.strip():
            raise ValueError(
                "Summary must not be empty."
            )

        generated_ids.append(
            citation_id
        )

    if generated_ids != expected_ids:
        raise ValueError(
            "Generated source IDs do not match "
            "PRRISM ranking."
        )

    # Build the public result deterministically from trusted evidence.
    ranked_results = []

    for rank, (
        source,
        generated,
    ) in enumerate(
        zip(
            evidence,
            generated_results,
        ),
        start=1,
    ):

        ranked_results.append(
            {
                "rank": rank,
                "citation_id": (
                    source[
                        "citation_id"
                    ]
                ),
                "source_id": (
                    source[
                        "source_id"
                    ]
                ),
                "title": (
                    source[
                        "title"
                    ]
                ),
                "url": (
                    source.get(
                        "url"
                    )
                ),
                "summary": (
                    generated[
                        "summary"
                    ].strip()
                ),
            }
        )

    return {
        "results": ranked_results
    }


# =============================================================================
# GENERATION
# =============================================================================

def generate_answer(
    question: str,
    evidence: List[Dict],
) -> Dict:
    """Generate readable explanations for ranked PRRISM evidence."""

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for "
            "answer generation."
        )

    prompt = build_generation_prompt(
        question,
        evidence,
    )

    client = Groq(
        api_key=api_key
    )

    response = (
        client
        .chat
        .completions
        .create(
            model=ANSWER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are PRRISM, a research "
                        "evidence assistant. Explain "
                        "each supplied source clearly "
                        "and independently. Never "
                        "change the supplied ranking "
                        "or combine evidence across "
                        "sources."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            response_format={
                "type": "json_object",
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:
        raise RuntimeError(
            "Answer model returned an empty "
            "response."
        )

    try:
        result = json.loads(
            content
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Answer model returned invalid JSON."
        ) from exc

    return validate_generated_answer(
        result,
        evidence,
    )


# =============================================================================
# CITATION / SOURCE METADATA
# =============================================================================

def build_citation_map(
    evidence: List[Dict],
) -> Dict[str, Dict]:
    """Map citation IDs to trusted PRRISM source metadata."""

    return {
        source["citation_id"]: {
            "source_id": (
                source[
                    "source_id"
                ]
            ),
            "title": (
                source[
                    "title"
                ]
            ),
            "url": (
                source.get(
                    "url"
                )
            ),
        }
        for source in evidence
    }