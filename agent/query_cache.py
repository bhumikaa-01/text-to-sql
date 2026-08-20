"""
Query result cache for the Text-to-SQL pipeline.

Caches successful final query responses so repeated questions
can avoid LLM calls and database execution.

The cache uses SQLite for persistence and is designed so the
backend can later be replaced with Redis for distributed deployments.
"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

# Project-root based cache path.
# Can be overridden with QUERY_CACHE_DB for deployment-specific storage.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DB = os.getenv(
    "QUERY_CACHE_DB",
    str(PROJECT_ROOT / "data" / "query_cache.db"),
)

CACHE_VERSION = "v2"

DEFAULT_TTL_SECONDS = 1800  # 30 minutes


def _normalize_question(question: str) -> str:
    """Normalize a natural-language question for stable cache keys."""

    return " ".join(
        question.strip().lower().split()
    )


def _make_cache_key(question: str) -> str:
    """Create a versioned deterministic SHA-256 cache key."""

    normalized = _normalize_question(question)

    key_material = (
        f"{CACHE_VERSION}:{normalized}"
    )

    return hashlib.sha256(
        key_material.encode("utf-8")
    ).hexdigest()


def _get_connection() -> sqlite3.Connection:
    """Create a SQLite connection for the cache."""

    connection = sqlite3.connect(
        CACHE_DB
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS query_cache (
            cache_key TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )

    connection.commit()

    return connection


def get_cached_response(
    question: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> dict[str, Any] | None:
    """
    Return a cached response if it exists and has not expired.

    Returns None on cache miss or expired entry.
    """

    cache_key = _make_cache_key(question)

    connection = _get_connection()

    try:
        row = connection.execute(
            """
            SELECT response_json, created_at
            FROM query_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()

        if row is None:
            return None

        response_json, created_at = row

        age = time.time() - created_at

        if age > ttl_seconds:

            connection.execute(
                """
                DELETE FROM query_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            )

            connection.commit()

            return None

        return json.loads(response_json)

    finally:
        connection.close()


def set_cached_response(
    question: str,
    response: dict[str, Any],
) -> None:
    """Store a successful final response in the cache."""

    cache_key = _make_cache_key(question)

    connection = _get_connection()

    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO query_cache (
                cache_key,
                question,
                response_json,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                cache_key,
                _normalize_question(question),
                json.dumps(response),
                time.time(),
            ),
        )

        connection.commit()

    finally:
        connection.close()


def clear_cache() -> None:
    """Remove all cached query responses."""

    connection = _get_connection()

    try:
        connection.execute(
            "DELETE FROM query_cache"
        )

        connection.commit()

    finally:
        connection.close()