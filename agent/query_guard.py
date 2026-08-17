"""
query_guard.py

Query resource and cost guardrails for generated SQL.

Purpose:
    Prevent technically valid but potentially expensive SQL
    from consuming excessive database resources.

Policy decisions:

    ALLOW
        Query is considered safe from a resource perspective.

    WARN
        Query is allowed to execute, but a resource concern is
        recorded for observability and future policy decisions.

    BLOCK
        Query exceeds a configured resource limit and must not
        reach the database.

This guard is deterministic and never executes SQL.
"""

from __future__ import annotations

import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum explicit LIMIT allowed.
MAX_LIMIT = 1000

# JOIN thresholds.
WARN_JOINS = 4
MAX_JOINS = 5

# UNION thresholds.
WARN_UNIONS = 3
MAX_UNIONS = 3


# ============================================================
# REGEX PATTERNS
# ============================================================

_LIMIT_PATTERN = re.compile(
    r"\bLIMIT\s+(\d+)",
    re.IGNORECASE,
)

_JOIN_PATTERN = re.compile(
    r"\bJOIN\b",
    re.IGNORECASE,
)

_UNION_PATTERN = re.compile(
    r"\bUNION\b",
    re.IGNORECASE,
)

_SELECT_STAR_PATTERN = re.compile(
    r"\bSELECT\s+\*",
    re.IGNORECASE,
)


# ============================================================
# RESULT HELPERS
# ============================================================

def _result(
    *,
    allowed: bool,
    decision: str,
    risk_level: str,
    reason: str = "",
    violations: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build a standardized resource-guard response.
    """

    return {
        "allowed": allowed,
        "decision": decision,
        "risk_level": risk_level,
        "reason": reason,
        "violations": violations or [],
    }


def _allow() -> dict[str, Any]:
    """
    Low-risk query.
    """

    return _result(
        allowed=True,
        decision="ALLOW",
        risk_level="LOW",
    )


def _warn(
    reason: str,
    violations: list[str],
) -> dict[str, Any]:
    """
    Medium-risk query.

    The query is allowed to execute, but the resource
    concern is surfaced for observability.
    """

    logger.warning(
        "Query resource guard warning: %s",
        reason,
    )

    return _result(
        allowed=True,
        decision="WARN",
        risk_level="MEDIUM",
        reason=reason,
        violations=violations,
    )


def _block(
    reason: str,
    violations: list[str],
) -> dict[str, Any]:
    """
    High-risk query.

    The query must not reach the database.
    """

    logger.warning(
        "Query resource guard blocked SQL: %s",
        reason,
    )

    return _result(
        allowed=False,
        decision="BLOCK",
        risk_level="HIGH",
        reason=reason,
        violations=violations,
    )


# ============================================================
# LIMIT ANALYSIS
# ============================================================

def _analyze_limit(
    sql: str,
) -> tuple[str | None, str | None]:
    """
    Analyze an explicit LIMIT.

    Returns:
        (decision, reason)

    decision:
        None   -> no LIMIT or acceptable LIMIT
        WARN   -> potentially large but acceptable
        BLOCK  -> exceeds hard maximum
    """

    match = _LIMIT_PATTERN.search(
        sql
    )

    if not match:
        return None, None

    limit_value = int(
        match.group(1)
    )

    if limit_value > MAX_LIMIT:

        return (
            "BLOCK",
            (
                f"LIMIT {limit_value} exceeds "
                f"maximum allowed LIMIT of {MAX_LIMIT}."
            ),
        )

    # LIMIT values above 500 are worth observing.
    if limit_value > 500:

        return (
            "WARN",
            (
                f"LIMIT {limit_value} is relatively large "
                "and may increase result-transfer cost."
            ),
        )

    return None, None


# ============================================================
# JOIN ANALYSIS
# ============================================================

def _analyze_joins(
    sql: str,
) -> tuple[str | None, str | None]:

    join_count = len(
        _JOIN_PATTERN.findall(sql)
    )

    if join_count > MAX_JOINS:

        return (
            "BLOCK",
            (
                f"Query contains {join_count} JOIN operations. "
                f"Maximum allowed is {MAX_JOINS}."
            ),
        )

    if join_count >= WARN_JOINS:

        return (
            "WARN",
            (
                f"Query contains {join_count} JOIN operations. "
                "Complex joins may increase query execution cost."
            ),
        )

    return None, None


# ============================================================
# UNION ANALYSIS
# ============================================================

def _analyze_unions(
    sql: str,
) -> tuple[str | None, str | None]:

    union_count = len(
        _UNION_PATTERN.findall(sql)
    )

    if union_count > MAX_UNIONS:

        return (
            "BLOCK",
            (
                f"Query contains {union_count} UNION operations. "
                f"Maximum allowed is {MAX_UNIONS}."
            ),
        )

    if union_count == MAX_UNIONS:

        return (
            "WARN",
            (
                f"Query contains {union_count} UNION operations. "
                "This is approaching the configured complexity limit."
            ),
        )

    return None, None


# ============================================================
# SELECT STAR ANALYSIS
# ============================================================

def _analyze_select_star(
    sql: str,
) -> tuple[str | None, str | None]:

    if not _SELECT_STAR_PATTERN.search(
        sql
    ):
        return None, None

    # SELECT * with a bounded LIMIT is allowed,
    # but we still expose it as a warning.
    if _LIMIT_PATTERN.search(sql):

        return (
            "WARN",
            (
                "SELECT * detected with a bounded LIMIT. "
                "Explicit columns are preferred to reduce "
                "data-transfer and serialization cost."
            ),
        )

    # SELECT * without LIMIT is a resource concern.
    return (
        "WARN",
        (
            "SELECT * detected without an explicit LIMIT. "
            "Explicit columns are preferred to reduce "
            "data-transfer and serialization cost."
        ),
    )


# ============================================================
# MAIN RESOURCE GUARD
# ============================================================

def check_query_resources(
    sql: str,
) -> dict[str, Any]:
    """
    Evaluate SQL for potentially expensive query patterns.

    This function:

        - never executes SQL
        - never modifies SQL
        - returns a deterministic policy decision

    Possible decisions:

        ALLOW
        WARN
        BLOCK
    """

    if not sql or not sql.strip():

        return _block(
            "Empty SQL cannot be evaluated.",
            [
                "EMPTY_SQL",
            ],
        )

    normalized_sql = sql.strip()

    warnings: list[str] = []
    warning_reasons: list[str] = []

    # ========================================================
    # 1. LIMIT
    # ========================================================

    limit_decision, limit_reason = (
        _analyze_limit(
            normalized_sql
        )
    )

    if limit_decision == "BLOCK":

        return _block(
            limit_reason or "Invalid LIMIT.",
            [
                "EXCESSIVE_LIMIT",
            ],
        )

    if limit_decision == "WARN":

        warnings.append(
            "LARGE_LIMIT"
        )

        if limit_reason:
            warning_reasons.append(
                limit_reason
            )

    # ========================================================
    # 2. JOIN complexity
    # ========================================================

    join_decision, join_reason = (
        _analyze_joins(
            normalized_sql
        )
    )

    if join_decision == "BLOCK":

        return _block(
            join_reason or "Too many JOIN operations.",
            [
                "EXCESSIVE_JOINS",
            ],
        )

    if join_decision == "WARN":

        warnings.append(
            "COMPLEX_JOINS"
        )

        if join_reason:
            warning_reasons.append(
                join_reason
            )

    # ========================================================
    # 3. UNION complexity
    # ========================================================

    union_decision, union_reason = (
        _analyze_unions(
            normalized_sql
        )
    )

    if union_decision == "BLOCK":

        return _block(
            union_reason or "Too many UNION operations.",
            [
                "EXCESSIVE_UNIONS",
            ],
        )

    if union_decision == "WARN":

        warnings.append(
            "COMPLEX_UNIONS"
        )

        if union_reason:
            warning_reasons.append(
                union_reason
            )

    # ========================================================
    # 4. SELECT *
    # ========================================================

    select_star_decision, select_star_reason = (
        _analyze_select_star(
            normalized_sql
        )
    )

    if select_star_decision == "WARN":

        warnings.append(
            "SELECT_STAR"
        )

        if select_star_reason:
            warning_reasons.append(
                select_star_reason
            )

    # ========================================================
    # 5. Final policy decision
    # ========================================================

    if warnings:

        return _warn(
            reason=" ".join(
                reason.strip()
                for reason in warning_reasons
                if reason
            ),
            violations=warnings,
        )

    return _allow()