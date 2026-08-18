"""
result_validator.py

Result and semantic validation for generated SQL.

This module evaluates whether an executed SQL result is
structurally healthy and whether the generated SQL contains
obvious semantic mismatches with the user's question.

Important:
    This is a validation layer, not a probability estimator.

    It produces observable validation signals that can later
    feed into the confidence / reliability engine.
"""

import re
from typing import Any


# ============================================================
# CONSTANTS
# ============================================================

PASS_SCORE = 100.0
WARN_SCORE = 60.0
FAIL_SCORE = 0.0

NULL_WARNING_THRESHOLD = 0.80


# ============================================================
# TEXT HELPERS
# ============================================================

def _contains_any(
    text: str,
    terms: tuple[str, ...],
) -> bool:
    """
    Return True if any term appears in the supplied text.
    """

    text = text.lower()

    return any(
        term in text
        for term in terms
    )


def _has_aggregate(
    sql: str,
) -> bool:
    """
    Check whether SQL contains a common aggregate function.
    """

    return bool(
        re.search(
            r"\b(SUM|AVG|COUNT|MIN|MAX)\s*\(",
            sql,
            flags=re.IGNORECASE,
        )
    )


def _has_group_by(
    sql: str,
) -> bool:
    """
    Check whether SQL contains GROUP BY.
    """

    return bool(
        re.search(
            r"\bGROUP\s+BY\b",
            sql,
            flags=re.IGNORECASE,
        )
    )


# ============================================================
# RESULT SHAPE VALIDATION
# ============================================================

def _validate_result_shape(
    results: Any,
) -> tuple[bool, str]:
    """
    Validate that execution returned the expected
    list-of-dictionaries structure.
    """

    if results is None:

        return (
            False,
            "Execution returned no result object.",
        )

    if not isinstance(
        results,
        list,
    ):

        return (
            False,
            "Query result is not a list.",
        )

    for row in results:

        if not isinstance(
            row,
            dict,
        ):

            return (
                False,
                "Query result contains a non-dictionary row.",
            )

    return (
        True,
        "",
    )


# ============================================================
# EMPTY RESULT CHECK
# ============================================================

def _check_empty_result(
    results: list[dict[str, Any]],
    sql: str,
) -> tuple[str, str]:
    """
    Determine whether an empty result should be treated
    as a warning.

    Empty results are not automatically incorrect because
    legitimate filters can return zero rows.
    """

    if results:

        return (
            "PASS",
            "",
        )

    if _contains_any(
        sql,
        (
            "COUNT(",
            "SUM(",
            "AVG(",
            "MIN(",
            "MAX(",
        ),
    ):

        return (
            "WARN",
            "Aggregate query returned no rows.",
        )

    return (
        "WARN",
        "Query returned no rows.",
    )


# ============================================================
# NULL DENSITY CHECK
# ============================================================

def _check_null_density(
    results: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Check whether an unusually large proportion of returned
    values are NULL.

    This is a result-health signal, not proof of semantic
    incorrectness.
    """

    if not results:

        return (
            "PASS",
            "",
        )

    total_values = 0
    null_values = 0

    for row in results:

        for value in row.values():

            total_values += 1

            if value is None:

                null_values += 1

    if total_values == 0:

        return (
            "PASS",
            "",
        )

    null_ratio = (
        null_values
        / total_values
    )

    if null_ratio >= NULL_WARNING_THRESHOLD:

        return (
            "WARN",
            (
                f"High NULL density detected "
                f"({null_ratio:.0%} of returned values)."
            ),
        )

    return (
        "PASS",
        "",
    )


# ============================================================
# BASIC SEMANTIC ALIGNMENT
# ============================================================

def _check_semantic_alignment(
    question: str,
    sql: str,
) -> tuple[str, str]:
    """
    Detect obvious semantic mismatches between the question
    and generated SQL.

    This deliberately does NOT claim that keyword checks can
    fully understand natural language.

    A later version can add an LLM-based semantic evaluator
    using the semantic layer as context.
    """

    question_lower = question.lower()
    sql_lower = sql.lower()

    # --------------------------------------------------------
    # Revenue / sales / GMV
    # --------------------------------------------------------

    if _contains_any(
        question_lower,
        (
            "revenue",
            "sales",
            "gmv",
        ),
    ):

        if "order_total_usd" not in sql_lower:

            return (
                "FAIL",
                (
                    "Revenue-related question does not "
                    "reference order_total_usd."
                ),
            )

        if "freight_value_usd" in sql_lower:

            return (
                "FAIL",
                (
                    "Revenue-related question uses "
                    "freight_value_usd as a revenue field."
                ),
            )

        if not _has_aggregate(sql):

            return (
                "WARN",
                (
                    "Revenue-related question does not "
                    "contain an aggregate operation."
                ),
            )

    # --------------------------------------------------------
    # Average
    # --------------------------------------------------------

    if _contains_any(
        question_lower,
        (
            "average",
            "avg",
            "mean",
        ),
    ):

        if not re.search(
            r"\bAVG\s*\(",
            sql,
            flags=re.IGNORECASE,
        ):

            return (
                "FAIL",
                (
                    "Question asks for an average but "
                    "SQL does not use AVG()."
                ),
            )

    # --------------------------------------------------------
    # Count
    # --------------------------------------------------------

    if _contains_any(
        question_lower,
        (
            "how many",
            "number of",
            "count",
        ),
    ):

        if not re.search(
            r"\bCOUNT\s*\(",
            sql,
            flags=re.IGNORECASE,
        ):

            return (
                "FAIL",
                (
                    "Question asks for a count but "
                    "SQL does not use COUNT()."
                ),
            )

    # --------------------------------------------------------
    # Monthly analysis
    # --------------------------------------------------------

    if _contains_any(
        question_lower,
        (
            "per month",
            "monthly",
            "by month",
            "month",
        ),
    ):

        if not _has_group_by(sql):

            return (
                "WARN",
                (
                    "Question requests monthly analysis "
                    "but SQL does not contain GROUP BY."
                ),
            )

    return (
        "PASS",
        "",
    )


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_query_result(
    *,
    question: str,
    generated_sql: str,
    results: Any,
) -> dict[str, Any]:
    """
    Validate an executed SQL result.

    Returns:

        {
            "status": "PASS" | "WARN" | "FAIL",
            "score": 0-100,
            "checks": {...},
            "issues": [...]
        }
    """

    checks: dict[str, Any] = {}
    issues: list[str] = []

    # ========================================================
    # Result shape
    # ========================================================

    shape_valid, shape_reason = (
        _validate_result_shape(
            results
        )
    )

    checks[
        "result_shape_valid"
    ] = shape_valid

    if not shape_valid:

        issues.append(
            shape_reason
        )

        return {
            "status": "FAIL",
            "score": FAIL_SCORE,
            "checks": checks,
            "issues": issues,
        }

    # ========================================================
    # Empty result
    # ========================================================

    empty_status, empty_reason = (
        _check_empty_result(
            results,
            generated_sql,
        )
    )

    checks[
        "has_results"
    ] = bool(results)

    if empty_status == "WARN":

        issues.append(
            empty_reason
        )

    # ========================================================
    # NULL density
    # ========================================================

    null_status, null_reason = (
        _check_null_density(
            results
        )
    )

    checks[
        "null_density_healthy"
    ] = (
        null_status == "PASS"
    )

    if null_status == "WARN":

        issues.append(
            null_reason
        )

    # ========================================================
    # Semantic alignment
    # ========================================================

    semantic_status, semantic_reason = (
        _check_semantic_alignment(
            question,
            generated_sql,
        )
    )

    checks[
        "semantic_alignment"
    ] = (
        semantic_status == "PASS"
    )

    if semantic_status != "PASS":

        issues.append(
            semantic_reason
        )

    # ========================================================
    # Final decision
    # ========================================================

    if semantic_status == "FAIL":

        status = "FAIL"
        score = FAIL_SCORE

    elif (
        empty_status == "WARN"
        or null_status == "WARN"
        or semantic_status == "WARN"
    ):

        status = "WARN"
        score = WARN_SCORE

    else:

        status = "PASS"
        score = PASS_SCORE

    return {
        "status": status,
        "score": score,
        "checks": checks,
        "issues": issues,
    }