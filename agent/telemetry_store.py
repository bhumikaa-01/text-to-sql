"""
Persistent telemetry storage for the Text-to-SQL observability system.

Stores QueryEvent records in a lightweight SQLite database so the
observability API and dashboard can retrieve historical query metrics.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agent.observability import QueryEvent


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DB_PATH = Path("data/observability.db")


# ============================================================
# DATABASE
# ============================================================

def _get_connection(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> sqlite3.Connection:
    """
    Create a SQLite connection and ensure the telemetry table exists.
    """

    path = Path(db_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        path,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS query_events (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            question TEXT,
            status TEXT,
            latency_ms INTEGER,

            sql_generated INTEGER,
            sql_safe INTEGER,

            sql_correction_attempted INTEGER,
            sql_correction_count INTEGER,
            sql_correction_applied INTEGER,

            cache_hit INTEGER,

            resource_decision TEXT,

            semantic_correct INTEGER,
            semantic_score REAL,

            confidence_score REAL,
            confidence_level TEXT,

            tables_used TEXT,

            error TEXT,

            metadata TEXT
        )
        """
    )

    connection.commit()

    return connection


# ============================================================
# RECORD EVENT
# ============================================================

def record_event(
    event: QueryEvent,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """
    Persist one QueryEvent.
    """

    connection = _get_connection(
        db_path
    )

    try:

        connection.execute(
            """
            INSERT OR REPLACE INTO query_events (
                request_id,
                timestamp,
                question,
                status,
                latency_ms,

                sql_generated,
                sql_safe,

                sql_correction_attempted,
                sql_correction_count,
                sql_correction_applied,

                cache_hit,

                resource_decision,

                semantic_correct,
                semantic_score,

                confidence_score,
                confidence_level,

                tables_used,

                error,

                metadata
            )
            VALUES (
                ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?,
                ?,
                ?, ?,
                ?, ?,
                ?,
                ?,
                ?
            )
            """,
            (
                event.request_id,
                event.timestamp,
                event.question,
                event.status,
                event.latency_ms,

                int(event.sql_generated),
                int(event.sql_safe),

                int(event.sql_correction_attempted),
                event.sql_correction_count,
                int(event.sql_correction_applied),

                int(event.cache_hit),

                event.resource_decision,

                (
                    None
                    if event.semantic_correct is None
                    else int(event.semantic_correct)
                ),
                event.semantic_score,

                event.confidence_score,
                event.confidence_level,

                json.dumps(event.tables_used),

                event.error,

                json.dumps(event.metadata),
            ),
        )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# RECENT EVENTS
# ============================================================

def get_recent_events(
    limit: int = 50,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """
    Return the most recent telemetry events.
    """

    if limit <= 0:
        return []

    connection = _get_connection(
        db_path
    )

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM query_events
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        return [
            _row_to_dict(row)
            for row in rows
        ]

    finally:

        connection.close()


# ============================================================
# SUMMARY METRICS
# ============================================================

def get_summary(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """
    Calculate aggregate observability metrics.
    """

    connection = _get_connection(
        db_path
    )

    try:

        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_queries,

                SUM(
                    CASE
                        WHEN status = 'SUCCESS'
                        THEN 1
                        ELSE 0
                    END
                ) AS successful_queries,

                SUM(
                    CASE
                        WHEN status = 'FAILED'
                        THEN 1
                        ELSE 0
                    END
                ) AS failed_queries,

                SUM(
                    CASE
                        WHEN status = 'BLOCKED'
                        THEN 1
                        ELSE 0
                    END
                ) AS blocked_queries,

                AVG(latency_ms) AS avg_latency_ms,

                AVG(confidence_score) AS avg_confidence_score,

                AVG(semantic_score) AS avg_semantic_score,

                SUM(
                    CASE
                        WHEN cache_hit = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS cache_hits,

                SUM(
                    CASE
                        WHEN sql_correction_attempted = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS correction_attempts,

                SUM(
                    CASE
                        WHEN sql_correction_applied = 1
                        THEN 1
                        ELSE 0
                    END
                ) AS corrections_applied

            FROM query_events
            """
        ).fetchone()

        total = row["total_queries"] or 0

        successful = (
            row["successful_queries"]
            or 0
        )

        cache_hits = (
            row["cache_hits"]
            or 0
        )

        correction_attempts = (
            row["correction_attempts"]
            or 0
        )

        corrections_applied = (
            row["corrections_applied"]
            or 0
        )

        return {
            "total_queries": total,

            "successful_queries": successful,

            "failed_queries": (
                row["failed_queries"]
                or 0
            ),

            "blocked_queries": (
                row["blocked_queries"]
                or 0
            ),

            "success_rate": (
                successful / total
                if total
                else 0.0
            ),

            "avg_latency_ms": (
                float(row["avg_latency_ms"])
                if row["avg_latency_ms"] is not None
                else 0.0
            ),

            "avg_confidence_score": (
                float(row["avg_confidence_score"])
                if row["avg_confidence_score"] is not None
                else 0.0
            ),

            "avg_semantic_score": (
                float(row["avg_semantic_score"])
                if row["avg_semantic_score"] is not None
                else 0.0
            ),

            "cache_hits": cache_hits,

            "cache_hit_rate": (
                cache_hits / total
                if total
                else 0.0
            ),

            "correction_attempts": correction_attempts,

            "corrections_applied": corrections_applied,

            "correction_rate": (
                correction_attempts / total
                if total
                else 0.0
            ),

            "correction_success_rate": (
                corrections_applied / correction_attempts
                if correction_attempts
                else 0.0
            ),
        }

    finally:

        connection.close()


# ============================================================
# ROW CONVERSION
# ============================================================

def _row_to_dict(
    row: sqlite3.Row,
) -> dict[str, Any]:
    """
    Convert a SQLite row back into the QueryEvent-style structure.
    """

    result = dict(row)

    result["sql_generated"] = bool(
        result["sql_generated"]
    )

    result["sql_safe"] = bool(
        result["sql_safe"]
    )

    result["sql_correction_attempted"] = bool(
        result["sql_correction_attempted"]
    )

    result["sql_correction_applied"] = bool(
        result["sql_correction_applied"]
    )

    result["cache_hit"] = bool(
        result["cache_hit"]
    )

    if result["semantic_correct"] is not None:

        result["semantic_correct"] = bool(
            result["semantic_correct"]
        )

    result["tables_used"] = json.loads(
        result["tables_used"]
        or "[]"
    )

    result["metadata"] = json.loads(
        result["metadata"]
        or "{}"
    )

    return result