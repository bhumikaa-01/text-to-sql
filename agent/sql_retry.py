"""
SQL retry and correction helpers for the Text-to-SQL pipeline.

Feature 7:
    Automatically retry SQL generation when the generated SQL
    fails deterministic validation such as schema validation.

The retry layer is intentionally bounded to prevent:
    - infinite LLM retry loops
    - unnecessary API costs
    - retry storms

Safety and resource violations are not automatically retried.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SQL_RETRIES = 2


# ============================================================
# RETRY POLICY
# ============================================================

RETRYABLE_ERROR_MARKERS = (
    "unknown column",
    "unknown table",
    "no such column",
    "no such table",
    "schema validation",
    "invalid sql",
    "syntax error",
    "sql parsing",
    "parse error",
)


NON_RETRYABLE_ERROR_MARKERS = (
    "sql safety",
    "unsafe sql",
    "resource guard",
    "excessive limit",
    "excessive joins",
    "excessive unions",
    "permission denied",
    "rate limit",
    "429",
    "503",
    "service unavailable",
)


def is_retryable_error(
    error: str | None,
) -> bool:
    """
    Determine whether an SQL-generation failure should trigger
    an automatic LLM correction attempt.

    Safety, resource, and infrastructure failures are explicitly
    excluded from automatic retries.
    """

    if not error:
        return False

    normalized_error = error.strip().lower()

    # --------------------------------------------------------
    # Safety / infrastructure failures always win.
    # --------------------------------------------------------

    if any(
        marker in normalized_error
        for marker in NON_RETRYABLE_ERROR_MARKERS
    ):
        return False

    # --------------------------------------------------------
    # Deterministic SQL/schema failures may be corrected.
    # --------------------------------------------------------

    return any(
        marker in normalized_error
        for marker in RETRYABLE_ERROR_MARKERS
    )


# ============================================================
# CORRECTION PROMPT
# ============================================================

def build_correction_prompt(
    *,
    question: str,
    previous_sql: str,
    validation_error: str,
    schema_context: str,
) -> str:
    """
    Build a focused correction prompt for the LLM.

    The model receives the original question, failed SQL,
    validation error, and relevant schema so it can correct
    the SQL instead of generating an unrelated query.
    """

    return f"""
You generated SQL for the following user question:

USER QUESTION:
{question}

Your previous SQL was:

PREVIOUS SQL:
{previous_sql}

The SQL failed deterministic validation.

VALIDATION ERROR:
{validation_error}

Use the following database schema:

SCHEMA:
{schema_context}

Correct the SQL so that it answers the original user question
and uses only valid tables and columns from the provided schema.

Important requirements:
- Return SQL only.
- Do not explain the correction.
- Do not use tables or columns that are not present in the schema.
- Preserve the original intent of the question.
- Do not introduce INSERT, UPDATE, DELETE, DROP, ALTER, or other
  destructive operations.
""".strip()


# ============================================================
# RETRY METADATA
# ============================================================

def build_retry_metadata(
    *,
    attempt: int,
    max_retries: int = MAX_SQL_RETRIES,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Build structured metadata for logging and observability.
    """

    return {
        "attempt": attempt,
        "max_retries": max_retries,
        "retry_available": attempt <= max_retries,
        "error": error or "",
    }