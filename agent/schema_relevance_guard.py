"""
schema_relevance_guard.py

Early deterministic guard for detecting questions that are clearly
outside the available database/schema scope.

This guard runs BEFORE RAG + LLM generation.

It does not determine whether a question is semantically answerable.
It only answers:

    "Does this question contain meaningful concepts that belong
     to our available database schema?"
"""

import re
from typing import Any

from agent.semantic_layer import SEMANTIC_SCHEMA


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


# ============================================================
# DOMAIN VOCABULARY
# ============================================================

def _build_schema_vocabulary() -> set[str]:
    """
    Build meaningful vocabulary from the semantic schema.

    Includes:
        - table names
        - column names
        - words from table descriptions
        - words from column descriptions
    """

    vocabulary: set[str] = set()

    for table in SEMANTIC_SCHEMA:

        table_name = table.get(
            "table_name",
            "",
        )

        vocabulary.update(
            _tokenize(table_name)
        )

        for key in (
            "description",
            "business_description",
        ):

            vocabulary.update(
                _tokenize(
                    table.get(key, "")
                )
            )

        for column in table.get(
            "columns",
            [],
        ):

            vocabulary.update(
                _tokenize(
                    column.get("name", "")
                )
            )

            for key in (
                "description",
                "business_description",
            ):

                vocabulary.update(
                    _tokenize(
                        column.get(key, "")
                    )
                )

    return {
        token
        for token in vocabulary
        if token not in STOPWORDS
        and len(token) >= 3
    }


def _tokenize(text: str) -> set[str]:
    """
    Convert text into normalized tokens.
    """

    if not text:
        return set()

    return set(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


SCHEMA_VOCABULARY = _build_schema_vocabulary()


# ============================================================
# RELEVANCE CHECK
# ============================================================

def check_schema_relevance(
    question: str,
) -> dict[str, Any]:
    """
    Deterministically check whether a question contains concepts
    represented by the available semantic schema.

    This is intentionally conservative.

    It should reject clearly unrelated questions such as:

        "What is the capital of France?"

    but allow legitimate database questions such as:

        "What was the average order value in 2018?"

    Returns
    -------
    dict
        {
            "relevant": bool,
            "matched_terms": [...],
            "question_terms": [...],
            "reason": str
        }
    """

    if not question or not question.strip():

        return {
            "relevant": False,
            "matched_terms": [],
            "question_terms": [],
            "reason": "EMPTY_QUESTION",
        }

    question_terms = _tokenize(question)

    meaningful_terms = {
        token
        for token in question_terms
        if token not in STOPWORDS
        and len(token) >= 3
    }

    matched_terms = sorted(
        meaningful_terms
        & SCHEMA_VOCABULARY
    )

    relevant = len(matched_terms) > 0

    if relevant:

        reason = (
            "Question contains concepts represented "
            "in the available database schema."
        )

    else:

        reason = (
            "Question does not contain meaningful "
            "concepts represented in the available "
            "database schema."
        )

    return {
        "relevant": relevant,
        "matched_terms": matched_terms,
        "question_terms": sorted(
            meaningful_terms
        ),
        "reason": reason,
    }