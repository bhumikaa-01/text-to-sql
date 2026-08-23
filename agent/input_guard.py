"""
input_guard.py

Deterministic security guard for raw user questions.

This guard runs BEFORE schema retrieval and LLM SQL generation.

Responsibilities:
    - Detect explicit destructive SQL requests.
    - Detect destructive operations expressed in natural language.
    - Detect SQL statement chaining.
    - Detect SQL comments commonly used in injection attempts.
    - Detect dangerous SQLite operations.
    - Prevent the LLM from silently ignoring malicious SQL
      embedded inside an otherwise valid natural-language request.

Important:
    This guard protects the system before SQL generation.
    It does NOT replace hitl_guard.py.

Security architecture:

    User Question
          |
          v
    Input Guard
          |
          v
    LLM SQL Generation
          |
          v
    HITL SQL Guard
          |
          v
    Resource Guard
          |
          v
    Database
"""

import logging
import re

logger = logging.getLogger(__name__)


# ============================================================
# DESTRUCTIVE SQL / NATURAL-LANGUAGE PATTERNS
# ============================================================

_DESTRUCTIVE_PATTERNS: list[
    tuple[re.Pattern[str], str, str]
] = [

    # --------------------------------------------------------
    # DROP
    # --------------------------------------------------------

    (
        re.compile(
            r"\bDROP\s+(?:TABLE|VIEW|INDEX|TRIGGER|DATABASE)\b",
            re.IGNORECASE,
        ),
        "DROP",
        "DROP operation detected in user input.",
    ),

    # Natural-language DROP request
    (
        re.compile(
            r"\b(?:DROP|REMOVE|DESTROY|DELETE)\s+"
            r"(?:THE\s+)?(?:TABLE|DATABASE|VIEW|INDEX)\b",
            re.IGNORECASE,
        ),
        "DROP",
        "Destructive database object operation detected in user input.",
    ),

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    # Explicit SQL:
    # DELETE FROM fact_orders
    (
        re.compile(
            r"\bDELETE\s+FROM\b",
            re.IGNORECASE,
        ),
        "DELETE",
        "DELETE operation detected in user input.",
    ),

    # Natural language:
    # Delete all canceled orders
    # Delete canceled orders
    # Remove all canceled orders
    # Remove the canceled orders
    (
        re.compile(
            r"\b(?:DELETE|REMOVE|ERASE|PURGE)\b"
            r"(?:\s+(?:ALL|THE|ANY))?"
            r"\s+(?:[A-Za-z0-9_*'\"_-]+\s*){1,20}"
            r"\b(?:ORDERS?|RECORDS?|ROWS?|DATA|ENTRIES|CUSTOMERS?|PRODUCTS?|"
            r"TRANSACTIONS?|REVIEWS?)\b",
            re.IGNORECASE,
        ),
        "DELETE",
        "Destructive data deletion request detected in user input.",
    ),

    # Generic destructive natural-language requests
    (
        re.compile(
            r"\b(?:DELETE|REMOVE|ERASE|PURGE)\s+"
            r"(?:ALL|THE|EVERY|ANY)\b",
            re.IGNORECASE,
        ),
        "DELETE",
        "Destructive data deletion request detected in user input.",
    ),

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    # Explicit SQL:
    # UPDATE table SET ...
    (
        re.compile(
            r"\bUPDATE\s+[A-Za-z_][A-Za-z0-9_]*\s+SET\b",
            re.IGNORECASE,
        ),
        "UPDATE",
        "UPDATE operation detected in user input.",
    ),

    # Natural-language update requests
    (
        re.compile(
            r"\b(?:UPDATE|MODIFY|CHANGE|ALTER)\b"
            r".*\b(?:SET|CHANGE|MODIFY|UPDATE)\b",
            re.IGNORECASE,
        ),
        "UPDATE",
        "Potential data modification request detected in user input.",
    ),

    # --------------------------------------------------------
    # INSERT
    # --------------------------------------------------------

    (
        re.compile(
            r"\bINSERT\s+INTO\b",
            re.IGNORECASE,
        ),
        "INSERT",
        "INSERT operation detected in user input.",
    ),

    # Natural-language insertion
    (
        re.compile(
            r"\b(?:INSERT|ADD)\s+"
            r"(?:A\s+|AN\s+|THE\s+|NEW\s+)?"
            r"(?:ROW|RECORD|DATA|ENTRY|CUSTOMER|PRODUCT|ORDER)\b",
            re.IGNORECASE,
        ),
        "INSERT",
        "Data insertion request detected in user input.",
    ),

    # --------------------------------------------------------
    # TRUNCATE
    # --------------------------------------------------------

    (
        re.compile(
            r"\bTRUNCATE\s+(?:TABLE\s+)?"
            r"[A-Za-z_][A-Za-z0-9_]*\b",
            re.IGNORECASE,
        ),
        "TRUNCATE",
        "TRUNCATE operation detected in user input.",
    ),

    (
        re.compile(
            r"\bTRUNCATE\b",
            re.IGNORECASE,
        ),
        "TRUNCATE",
        "TRUNCATE operation detected in user input.",
    ),

    # --------------------------------------------------------
    # ALTER
    # --------------------------------------------------------

    (
        re.compile(
            r"\bALTER\s+(?:TABLE|DATABASE)\b",
            re.IGNORECASE,
        ),
        "ALTER",
        "ALTER operation detected in user input.",
    ),

    # --------------------------------------------------------
    # CREATE
    # --------------------------------------------------------

    (
        re.compile(
            r"\bCREATE\s+"
            r"(?:TABLE|DATABASE|VIEW|INDEX|TRIGGER)\b",
            re.IGNORECASE,
        ),
        "CREATE",
        "CREATE operation detected in user input.",
    ),

    # --------------------------------------------------------
    # ATTACH
    # --------------------------------------------------------

    (
        re.compile(
            r"\bATTACH\s+DATABASE\b",
            re.IGNORECASE,
        ),
        "ATTACH",
        "ATTACH DATABASE operation detected in user input.",
    ),

    # --------------------------------------------------------
    # DETACH
    # --------------------------------------------------------

    (
        re.compile(
            r"\bDETACH\s+DATABASE\b",
            re.IGNORECASE,
        ),
        "DETACH",
        "DETACH DATABASE operation detected in user input.",
    ),

    # --------------------------------------------------------
    # PRAGMA
    # --------------------------------------------------------

    (
        re.compile(
            r"\bPRAGMA(?:\s+[A-Za-z_][A-Za-z0-9_]*)?",
            re.IGNORECASE,
        ),
        "PRAGMA",
        "PRAGMA operation detected in user input.",
    ),

    # --------------------------------------------------------
    # VACUUM
    # --------------------------------------------------------

    (
        re.compile(
            r"\bVACUUM\b",
            re.IGNORECASE,
        ),
        "VACUUM",
        "VACUUM operation detected in user input.",
    ),

    # --------------------------------------------------------
    # REINDEX
    # --------------------------------------------------------

    (
        re.compile(
            r"\bREINDEX\b",
            re.IGNORECASE,
        ),
        "REINDEX",
        "REINDEX operation detected in user input.",
    ),

    # --------------------------------------------------------
    # GRANT / REVOKE
    # --------------------------------------------------------

    (
        re.compile(
            r"\bGRANT\b",
            re.IGNORECASE,
        ),
        "GRANT",
        "GRANT operation detected in user input.",
    ),

    (
        re.compile(
            r"\bREVOKE\b",
            re.IGNORECASE,
        ),
        "REVOKE",
        "REVOKE operation detected in user input.",
    ),
]


# ============================================================
# SQL INJECTION / STATEMENT CHAINING PATTERNS
# ============================================================

_INJECTION_PATTERNS: list[
    tuple[re.Pattern[str], str]
] = [

    # Example:
    # SELECT ...; DROP TABLE ...
    (
        re.compile(
            r";\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER|"
            r"TRUNCATE|CREATE|ATTACH|DETACH|PRAGMA|"
            r"VACUUM|REINDEX|GRANT|REVOKE)\b",
            re.IGNORECASE,
        ),
        "SQL statement chaining with a dangerous operation detected.",
    ),

    # Any semicolon followed by another SQL statement keyword
    (
        re.compile(
            r";\s*(?:SELECT|WITH|DROP|DELETE|UPDATE|INSERT|"
            r"ALTER|TRUNCATE|CREATE|ATTACH|DETACH|PRAGMA|"
            r"VACUUM|REINDEX)\b",
            re.IGNORECASE,
        ),
        "Multiple SQL statements detected in user input.",
    ),

    # SQL single-line comment
    (
        re.compile(
            r"--",
            re.IGNORECASE,
        ),
        "SQL single-line comment marker detected in user input.",
    ),

    # SQL block comment opening
    (
        re.compile(
            r"/\*",
            re.IGNORECASE,
        ),
        "SQL block comment marker detected in user input.",
    ),

    # SQL block comment closing
    (
        re.compile(
            r"\*/",
            re.IGNORECASE,
        ),
        "SQL block comment terminator detected in user input.",
    ),
]


# ============================================================
# HELPERS
# ============================================================

def _normalize_input(question: str) -> str:
    """
    Normalize whitespace without changing the meaning
    of the user's question.
    """

    return re.sub(
        r"\s+",
        " ",
        question.strip(),
    )


def _blocked_result(
    operation: str,
    reason: str,
) -> dict:
    """
    Return a standardized blocked security decision.
    """

    return {
        "allowed": False,
        "risk_level": "CRITICAL",
        "operation": operation,
        "reason": reason,
        "violations": [
            operation,
        ],
    }


# ============================================================
# PUBLIC API
# ============================================================

def check_user_input(
    question: str,
) -> dict:
    """
    Check raw user input before SQL generation.

    Returns:

        {
            "allowed": bool,
            "risk_level": str,
            "operation": str,
            "reason": str,
            "violations": list[str],
        }
    """

    # --------------------------------------------------------
    # 0. Empty input
    # --------------------------------------------------------

    if not question or not question.strip():

        return _blocked_result(
            "EMPTY_INPUT",
            "User question is empty.",
        )

    normalized_question = _normalize_input(
        question
    )

    # --------------------------------------------------------
    # 1. Detect explicit destructive operations
    # --------------------------------------------------------

    for (
        pattern,
        operation,
        reason,
    ) in _DESTRUCTIVE_PATTERNS:

        if pattern.search(
            normalized_question
        ):

            logger.warning(
                "INPUT GUARD blocked %s operation in user input: %s",
                operation,
                normalized_question,
            )

            return _blocked_result(
                operation,
                reason,
            )

    # --------------------------------------------------------
    # 2. Detect SQL injection / statement chaining
    # --------------------------------------------------------

    for (
        pattern,
        reason,
    ) in _INJECTION_PATTERNS:

        if pattern.search(
            normalized_question
        ):

            logger.warning(
                "INPUT GUARD blocked suspicious user input: %s",
                reason,
            )

            return _blocked_result(
                "INJECTION",
                reason,
            )

    # --------------------------------------------------------
    # 3. Safe natural-language request
    # --------------------------------------------------------

    return {
        "allowed": True,
        "risk_level": "LOW",
        "operation": "READ",
        "reason": "",
        "violations": [],
    }