"""
confidence.py

Confidence / reliability scoring for generated SQL.

The score is based on observable signals from the
Text-to-SQL pipeline rather than an additional LLM judgment.

Score:
    0 - 100

Levels:
    HIGH   : 90 - 100
    MEDIUM : 70 - 89
    LOW    : 0 - 69

Important:
    Some signals may be unavailable during live production
    queries because ground-truth information is not known.
    Such signals are treated as unverified rather than
    incorrectly assuming correctness.
"""

from typing import Any


# ============================================================
# SCORE WEIGHTS
# ============================================================

SQL_SAFETY_WEIGHT = 15
SCHEMA_VALIDITY_WEIGHT = 20
RESOURCE_SAFETY_WEIGHT = 10
EXECUTION_WEIGHT = 20
RESULT_QUALITY_WEIGHT = 25
TABLE_CORRECTNESS_WEIGHT = 10


# ============================================================
# SCORE LEVELS
# ============================================================

HIGH_THRESHOLD = 90
MEDIUM_THRESHOLD = 70


def _get_level(
    score: float,
) -> str:
    """
    Convert numeric confidence score into
    a human-readable reliability level.
    """

    if score >= HIGH_THRESHOLD:
        return "HIGH"

    if score >= MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "LOW"


def calculate_confidence(
    *,
    sql_safe: bool,
    schema_valid: bool,
    resource_decision: str,
    execution_success: bool,
    result_quality: float,
    table_correct: bool | None,
) -> dict[str, Any]:
    """
    Calculate a reliability score for a generated SQL query.

    Parameters
    ----------
    sql_safe:
        Whether the generated SQL passed the SQL safety guard.

    schema_valid:
        Whether the generated SQL passed schema validation.

    resource_decision:
        Resource guard decision:
            ALLOW
            WARN
            BLOCK

    execution_success:
        Whether the generated SQL executed successfully.

    result_quality:
        Result quality expressed as a value from 0 to 100.

        In live production mode this represents observable
        result health, not verified semantic correctness.

    table_correct:
        Whether generated tables match expected tables.

        True:
            Verified correct.

        False:
            Verified incorrect.

        None:
            Ground truth is unavailable, such as during a
            normal live production query.

    Returns
    -------
    dict[str, Any]
        Structured reliability information containing:

            score
            level
            factors
    """

    # ========================================================
    # Normalize inputs
    # ========================================================

    resource_decision = (
        resource_decision or "BLOCK"
    ).upper()

    result_quality = max(
        0.0,
        min(
            100.0,
            float(result_quality),
        ),
    )

    # ========================================================
    # SQL safety score
    # ========================================================

    sql_safety_score = (
        SQL_SAFETY_WEIGHT
        if sql_safe
        else 0.0
    )

    # ========================================================
    # Schema validity score
    # ========================================================

    schema_validity_score = (
        SCHEMA_VALIDITY_WEIGHT
        if schema_valid
        else 0.0
    )

    # ========================================================
    # Resource safety score
    # ========================================================
    #
    # ALLOW  → full points
    # WARN   → half points
    # BLOCK  → zero
    #

    if resource_decision == "ALLOW":

        resource_safety_score = (
            RESOURCE_SAFETY_WEIGHT
        )

    elif resource_decision == "WARN":

        resource_safety_score = (
            RESOURCE_SAFETY_WEIGHT
            * 0.5
        )

    else:

        resource_safety_score = 0.0

    # ========================================================
    # Execution score
    # ========================================================

    execution_score = (
        EXECUTION_WEIGHT
        if execution_success
        else 0.0
    )

    # ========================================================
    # Result quality score
    # ========================================================

    result_quality_score = (
        RESULT_QUALITY_WEIGHT
        * (
            result_quality
            / 100.0
        )
    )

    # ========================================================
    # Table correctness score
    # ========================================================
    #
    # None means that ground truth is unavailable.
    #
    # We deliberately do NOT award points in this case.
    # This prevents the production system from claiming
    # that it knows the generated tables are correct.
    #

    if table_correct is True:

        table_correctness_score = (
            TABLE_CORRECTNESS_WEIGHT
        )

    elif table_correct is False:

        table_correctness_score = 0.0

    else:

        table_correctness_score = 0.0

    # ========================================================
    # Base score
    # ========================================================

    raw_score = (
        sql_safety_score
        + schema_validity_score
        + resource_safety_score
        + execution_score
        + result_quality_score
        + table_correctness_score
    )

    score = raw_score

    # ========================================================
    # Hard-failure caps
    # ========================================================
    #
    # These prevent a query with a serious failure from
    # receiving an artificially high confidence score.
    #

    if resource_decision == "BLOCK":

        score = 0.0

    elif not sql_safe:

        score = min(
            score,
            20.0,
        )

    elif not schema_valid:

        score = min(
            score,
            30.0,
        )

    elif not execution_success:

        score = min(
            score,
            40.0,
        )

    # ========================================================
    # Final normalization
    # ========================================================

    score = round(
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        2,
    )

    level = _get_level(
        score
    )

    # ========================================================
    # Structured response
    # ========================================================

    return {
        "score": score,

        "level": level,

        "factors": {
            "sql_safety": round(
                sql_safety_score,
                2,
            ),

            "schema_validity": round(
                schema_validity_score,
                2,
            ),

            "resource_safety": round(
                resource_safety_score,
                2,
            ),

            "execution": round(
                execution_score,
                2,
            ),

            "result_quality": round(
                result_quality_score,
                2,
            ),

            "table_correctness": round(
                table_correctness_score,
                2,
            ),
        },
    }