"""
semantic_evaluator.py

LLM-based semantic correctness evaluator for Text-to-SQL.

The evaluator receives:
    - user question
    - generated SQL
    - actual query result
    - semantic schema

It evaluates whether the generated SQL and returned result
actually answer the user's question.

Important:
    This evaluator does NOT:
        - execute SQL
        - modify the database
        - perform SQL safety checks
        - perform resource checks

It is only responsible for semantic correctness.
"""

import json
import logging
from typing import Any

from google import genai

from agent.semantic_layer import SEMANTIC_SCHEMA


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-2.5-flash-lite"


# ============================================================
# CLIENT
# ============================================================

_client = genai.Client()


# ============================================================
# SEMANTIC SCHEMA CONTEXT
# ============================================================

def _build_schema_context() -> str:
    """
    Convert the semantic schema into compact JSON context
    for the evaluator.
    """

    return json.dumps(
        SEMANTIC_SCHEMA,
        indent=2,
    )


# ============================================================
# EVALUATION PROMPT
# ============================================================

def _build_prompt(
    *,
    question: str,
    generated_sql: str,
    results: list[dict[str, Any]],
) -> str:
    """
    Build a grounded semantic evaluation prompt.
    """

    schema_context = (
        _build_schema_context()
    )

    result_context = json.dumps(
        results,
        indent=2,
        default=str,
    )

    return f"""
You are a semantic correctness evaluator for a
Text-to-SQL system.

Your ONLY task is to determine whether the generated SQL
and its actual result correctly answer the user's question.

Do NOT execute SQL.

Do NOT modify SQL.

Do NOT evaluate SQL security.

Do NOT evaluate query cost.

Use ONLY:
    1. The user's question
    2. The generated SQL
    3. The actual query result
    4. The provided semantic schema

------------------------------------------------------------
SEMANTIC SCHEMA
------------------------------------------------------------

{schema_context}

------------------------------------------------------------
USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
GENERATED SQL
------------------------------------------------------------

{generated_sql}

------------------------------------------------------------
ACTUAL QUERY RESULT
------------------------------------------------------------

{result_context}

------------------------------------------------------------
EVALUATION CRITERIA
------------------------------------------------------------

Check:

1. Does the SQL address the user's actual intent?

2. Are the selected columns and tables appropriate?

3. Are aggregation functions appropriate?

4. Are filters appropriate?

5. Are GROUP BY operations appropriate when required?

6. Does the actual result have the expected meaning
   for the question?

7. Does the SQL follow the business definitions in
   the semantic schema?

Be conservative.

If the SQL is executable but answers a different question,
mark it incorrect.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Return ONLY valid JSON.

Use exactly this structure:

{{
    "is_correct": true,
    "score": 0.95,
    "reason": "Brief explanation.",
    "issues": []
}}

Rules:

- is_correct must be boolean.
- score must be between 0.0 and 1.0.
- reason must be a concise explanation.
- issues must be a JSON list of strings.
- Do not include markdown.
- Do not include ``` fences.
"""


# ============================================================
# RESPONSE PARSING
# ============================================================

def _parse_response(
    response_text: str,
) -> dict[str, Any]:
    """
    Parse and validate the evaluator's JSON response.
    """

    text = (
        response_text
        .strip()
    )

    # --------------------------------------------------------
    # Remove accidental markdown fences
    # --------------------------------------------------------

    if text.startswith(
        "```"
    ):

        text = (
            text.replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            text
        )

    except json.JSONDecodeError as exc:

        logger.error(
            "Semantic evaluator returned invalid JSON: %s",
            response_text,
        )

        raise ValueError(
            "Semantic evaluator returned invalid JSON."
        ) from exc

    # --------------------------------------------------------
    # Validate required fields
    # --------------------------------------------------------

    required_fields = (
        "is_correct",
        "score",
        "reason",
        "issues",
    )

    for field in required_fields:

        if field not in data:

            raise ValueError(
                (
                    "Semantic evaluator response "
                    f"is missing field: {field}"
                )
            )

    # --------------------------------------------------------
    # Normalize fields
    # --------------------------------------------------------

    is_correct = data[
        "is_correct"
    ]

    if not isinstance(
        is_correct,
        bool,
    ):

        raise ValueError(
            "is_correct must be a boolean."
        )

    try:

        score = float(
            data["score"]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "score must be numeric."
        ) from exc

    score = max(
        0.0,
        min(
            1.0,
            score,
        ),
    )

    reason = str(
        data["reason"]
    )

    issues = data[
        "issues"
    ]

    if not isinstance(
        issues,
        list,
    ):

        raise ValueError(
            "issues must be a list."
        )

    issues = [
        str(issue)
        for issue in issues
    ]

    return {
        "is_correct": is_correct,
        "score": round(
            score,
            4,
        ),
        "reason": reason,
        "issues": issues,
    }


# ============================================================
# PUBLIC EVALUATOR
# ============================================================

def evaluate_semantics(
    *,
    question: str,
    generated_sql: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate semantic correctness using Gemini.

    Returns:

        {
            "is_correct": bool,
            "score": float,
            "reason": str,
            "issues": list[str]
        }
    """

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not generated_sql.strip():

        return {
            "is_correct": False,
            "score": 0.0,
            "reason": (
                "No SQL was generated for evaluation."
            ),
            "issues": [
                "EMPTY_SQL"
            ],
        }

    prompt = _build_prompt(
        question=question,
        generated_sql=generated_sql,
        results=results,
    )

    # --------------------------------------------------------
    # Call evaluator model
    # --------------------------------------------------------

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    response_text = (
        response.text
        or ""
    )

    logger.info(
        "Semantic evaluator raw response: %s",
        response_text,
    )

    if not response_text.strip():

        raise ValueError(
            "Semantic evaluator returned an empty response."
        )

    # --------------------------------------------------------
    # Parse structured response
    # --------------------------------------------------------

    return _parse_response(
        response_text
    )