"""
sql_corrector.py

LLM-based SQL correction for Text-to-SQL.

The corrector receives:
    - user question
    - generated SQL
    - semantic evaluation
    - semantic schema

It produces a corrected SQL query.

Important:
    This module does NOT:
        - execute SQL
        - bypass SQL safety checks
        - bypass resource checks
        - modify the database

The corrected SQL must be validated by the
existing SQL safety and resource guards before execution.
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
# SCHEMA CONTEXT
# ============================================================

def _build_schema_context() -> str:
    """
    Convert semantic schema into compact JSON context.
    """

    return json.dumps(
        SEMANTIC_SCHEMA,
        indent=2,
    )


# ============================================================
# CORRECTION PROMPT
# ============================================================

def _build_prompt(
    *,
    question: str,
    generated_sql: str,
    evaluation: dict[str, Any],
) -> str:
    """
    Build a grounded SQL correction prompt.
    """

    schema_context = _build_schema_context()

    evaluation_context = json.dumps(
        evaluation,
        indent=2,
        default=str,
    )

    return f"""
You are a SQL correction agent for a Text-to-SQL system.

Your task is to correct an SQL query that was determined
to be semantically incorrect.

You MUST preserve the user's original intent.

Do NOT execute SQL.

Do NOT modify the database.

Do NOT generate destructive SQL.

Return ONLY the corrected SQL query.

------------------------------------------------------------
SEMANTIC SCHEMA
------------------------------------------------------------

{schema_context}

------------------------------------------------------------
USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ORIGINAL SQL
------------------------------------------------------------

{generated_sql}

------------------------------------------------------------
SEMANTIC EVALUATION
------------------------------------------------------------

{evaluation_context}

------------------------------------------------------------
CORRECTION RULES
------------------------------------------------------------

1. Correct the semantic problems identified by the evaluator.

2. Preserve the original user intent.

3. Use only tables and columns supported by the semantic schema.

4. Use the correct aggregation:
   SUM for total revenue,
   COUNT for counts,
   AVG for averages,
   etc.

5. Preserve required filters.

6. Preserve required grouping.

7. Preserve ranking and LIMIT requirements.

8. Do not invent tables or columns.

9. Do not generate:
   INSERT
   UPDATE
   DELETE
   DROP
   ALTER
   TRUNCATE
   CREATE

10. Return ONLY SQL.

------------------------------------------------------------
CORRECTED SQL
------------------------------------------------------------
"""


# ============================================================
# RESPONSE CLEANING
# ============================================================

def _clean_sql_response(
    response_text: str,
) -> str:
    """
    Clean accidental markdown formatting from
    the LLM response.
    """

    sql = response_text.strip()

    if sql.startswith("```"):

        sql = (
            sql.replace("```sql", "")
            .replace("```", "")
            .strip()
        )

    if not sql:
        raise ValueError(
            "SQL correction model returned empty SQL."
        )

    return sql


# ============================================================
# PUBLIC CORRECTOR
# ============================================================

def correct_sql(
    *,
    question: str,
    generated_sql: str,
    evaluation: dict[str, Any],
) -> str:
    """
    Correct semantically incorrect SQL.

    Returns:
        Corrected SQL string.
    """

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not generated_sql.strip():

        raise ValueError(
            "Generated SQL cannot be empty."
        )

    if evaluation.get("is_correct") is True:

        return generated_sql.strip()

    prompt = _build_prompt(
        question=question,
        generated_sql=generated_sql,
        evaluation=evaluation,
    )

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    response_text = response.text or ""

    logger.info(
        "SQL corrector raw response: %s",
        response_text,
    )

    return _clean_sql_response(
        response_text
    )