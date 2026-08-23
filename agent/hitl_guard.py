"""
hitl_guard.py

Production SQL safety guard for generated SQL.

The guard performs deterministic checks before SQL reaches
the database execution layer.

Responsibilities:
    - Allow safe read-only SQL.
    - Detect destructive/write operations.
    - Detect multiple SQL statements.
    - Detect dangerous SQLite operations.
    - Detect suspicious SQL comment/injection patterns.
    - Return a structured safety decision.

Important:
    This layer should never trust the LLM-generated SQL.
"""

import logging
import re


logger = logging.getLogger(__name__)


# ============================================================
# SQL OPERATION PATTERNS
# ============================================================

_DANGEROUS_PATTERNS: list[
    tuple[re.Pattern[str], str, str, str]
] = [

    (
        re.compile(
            r"\bINSERT\s+INTO\b",
            re.IGNORECASE,
        ),
        "INSERT",
        "CRITICAL",
        "INSERT statement detected — would modify database data.",
    ),

    (
        re.compile(
            r"\bUPDATE\b",
            re.IGNORECASE,
        ),
        "UPDATE",
        "CRITICAL",
        "UPDATE statement detected — would modify database data.",
    ),

    (
        re.compile(
            r"\bDELETE\s+FROM\b",
            re.IGNORECASE,
        ),
        "DELETE",
        "CRITICAL",
        "DELETE statement detected — would remove database data.",
    ),

    (
        re.compile(
            r"\bDROP\b",
            re.IGNORECASE,
        ),
        "DROP",
        "CRITICAL",
        "DROP statement detected — could destroy database objects.",
    ),

    (
        re.compile(
            r"\bTRUNCATE\b",
            re.IGNORECASE,
        ),
        "TRUNCATE",
        "CRITICAL",
        "TRUNCATE statement detected — would remove database data.",
    ),

    (
        re.compile(
            r"\bALTER\b",
            re.IGNORECASE,
        ),
        "ALTER",
        "CRITICAL",
        "ALTER statement detected — would modify database schema.",
    ),

    (
        re.compile(
            r"\bCREATE\b",
            re.IGNORECASE,
        ),
        "CREATE",
        "CRITICAL",
        "CREATE statement detected — would modify database schema.",
    ),

    (
        re.compile(
            r"\bGRANT\b",
            re.IGNORECASE,
        ),
        "GRANT",
        "CRITICAL",
        "GRANT statement detected — would modify permissions.",
    ),

    (
        re.compile(
            r"\bREVOKE\b",
            re.IGNORECASE,
        ),
        "REVOKE",
        "CRITICAL",
        "REVOKE statement detected — would modify permissions.",
    ),

    (
        re.compile(
            r"\bATTACH\s+DATABASE\b",
            re.IGNORECASE,
        ),
        "ATTACH",
        "CRITICAL",
        "ATTACH DATABASE detected — could access an external database.",
    ),

    (
        re.compile(
            r"\bDETACH\s+DATABASE\b",
            re.IGNORECASE,
        ),
        "DETACH",
        "CRITICAL",
        "DETACH DATABASE detected.",
    ),

    (
        re.compile(
            r"\bPRAGMA\b",
            re.IGNORECASE,
        ),
        "PRAGMA",
        "HIGH",
        "PRAGMA statement detected — database configuration access is restricted.",
    ),

    (
        re.compile(
            r"\bVACUUM\b",
            re.IGNORECASE,
        ),
        "VACUUM",
        "HIGH",
        "VACUUM operation is not permitted through generated SQL.",
    ),

    (
        re.compile(
            r"\bREINDEX\b",
            re.IGNORECASE,
        ),
        "REINDEX",
        "HIGH",
        "REINDEX operation is not permitted through generated SQL.",
    ),
]


# ============================================================
# SQL COMMENTS / INJECTION PATTERNS
# ============================================================

_INJECTION_PATTERNS: list[
    tuple[re.Pattern[str], str]
] = [

    (
        re.compile(
            r";\s*--",
            re.IGNORECASE,
        ),
        "Possible SQL injection comment terminator detected.",
    ),

    (
        re.compile(
            r";\s*/\*",
            re.IGNORECASE,
        ),
        "Possible SQL injection block comment detected.",
    ),
]


# ============================================================
# HELPERS
# ============================================================

def _base_result() -> dict:
    """
    Return the default safe result.
    """

    return {
        "allowed": True,
        "requires_approval": False,
        "risk_level": "LOW",
        "operation": "SELECT",
        "reason": "",
    }


def _normalize_sql(
    sql: str,
) -> str:
    """
    Normalize whitespace while preserving SQL semantics.
    """

    return re.sub(
        r"\s+",
        " ",
        sql.strip(),
    )


def _remove_sql_comments(
    sql: str,
) -> str:
    """
    Remove SQL comments before operation detection.

    This prevents natural-language model refusals such as:

        -- I cannot generate DELETE statements.

        -- INSERT, UPDATE, DELETE, or DROP are forbidden.

    from being interpreted as executable SQL operations.

    The actual SQL statement is still validated separately.
    """

    # Remove single-line SQL comments.
    sql = re.sub(
        r"--[^\n]*",
        " ",
        sql,
    )

    # Remove block SQL comments.
    sql = re.sub(
        r"/\*.*?\*/",
        " ",
        sql,
        flags=re.DOTALL,
    )

    return sql


# ============================================================
# PUBLIC API
# ============================================================

def check_sql(
    sql: str,
) -> dict:
    """
    Perform deterministic safety checks on generated SQL.

    Returns a structured decision:

        {
            "allowed": bool,
            "requires_approval": bool,
            "risk_level": str,
            "operation": str,
            "reason": str,
        }

    The guard never raises for normal validation failures.
    """

    # --------------------------------------------------------
    # 0. Empty SQL
    # --------------------------------------------------------

    if not sql or not sql.strip():

        return {
            "allowed": False,
            "requires_approval": False,
            "risk_level": "HIGH",
            "operation": "EMPTY",
            "reason": "SQL query is empty.",
        }

    normalized_sql = _normalize_sql(
        sql
    )

    # Keep the original normalized SQL for statement/type
    # validation, but remove comments for operation detection.
    sql_for_operation_detection = _remove_sql_comments(
        normalized_sql
    )

    # Normalize whitespace again after removing comments.
    sql_for_operation_detection = _normalize_sql(
        sql_for_operation_detection
    )

    # --------------------------------------------------------
    # 1. Multiple statement detection
    # --------------------------------------------------------

    statements = [
        statement.strip()
        for statement in normalized_sql.split(";")
        if statement.strip()
    ]

    if len(statements) > 1:

        logger.warning(
            "SQL guard blocked multiple SQL statements."
        )

        return {
            "allowed": False,
            "requires_approval": False,
            "risk_level": "CRITICAL",
            "operation": "MULTI_STATEMENT",
            "reason": (
                "Multiple SQL statements are not permitted."
            ),
        }

    # --------------------------------------------------------
    # 2. Dangerous operation detection
    # --------------------------------------------------------

    for (
        pattern,
        operation,
        risk_level,
        reason,
    ) in _DANGEROUS_PATTERNS:

        if pattern.search(
            sql_for_operation_detection
        ):

            logger.warning(
                "SQL guard blocked %s operation: %s",
                operation,
                reason,
            )

            return {
                "allowed": False,
                "requires_approval": False,
                "risk_level": risk_level,
                "operation": operation,
                "reason": reason,
            }

    # --------------------------------------------------------
    # 3. Injection / comment detection
    # --------------------------------------------------------

    for pattern, reason in _INJECTION_PATTERNS:

        if pattern.search(
            normalized_sql
        ):

            logger.warning(
                "SQL guard blocked suspicious SQL pattern: %s",
                reason,
            )

            return {
                "allowed": False,
                "requires_approval": False,
                "risk_level": "CRITICAL",
                "operation": "INJECTION",
                "reason": reason,
            }

    # --------------------------------------------------------
    # 4. Statement type validation
    # --------------------------------------------------------

    # Only SELECT and WITH are allowed.
    #
    # IMPORTANT:
    # We validate the comment-stripped SQL here.
    #
    # Example:
    #
    #   -- I cannot generate DELETE statements.
    #
    # becomes empty after comment removal and therefore
    # gets rejected as UNSUPPORTED instead of being detected
    # as an UPDATE/DELETE operation.
    #

    executable_sql = sql_for_operation_detection.strip()

    if not executable_sql:

        logger.warning(
            "SQL guard blocked SQL containing no executable statement."
        )

        return {
            "allowed": False,
            "requires_approval": False,
            "risk_level": "HIGH",
            "operation": "EMPTY",
            "reason": (
                "No executable SQL statement was generated."
            ),
        }

    if not re.match(
        r"^(SELECT|WITH)\b",
        executable_sql,
        re.IGNORECASE,
    ):

        logger.warning(
            "SQL guard blocked unsupported statement type."
        )

        return {
            "allowed": False,
            "requires_approval": False,
            "risk_level": "HIGH",
            "operation": "UNSUPPORTED",
            "reason": (
                "Only SELECT and WITH queries are allowed."
            ),
        }

    # --------------------------------------------------------
    # 5. Safe read-only query
    # --------------------------------------------------------

    return _base_result()