"""
Observability and telemetry helpers for the Text-to-SQL pipeline.

This module provides a structured representation of query execution
events so that the pipeline can later expose metrics through an
observability dashboard, API, evaluation system, or monitoring layer.

The module intentionally does not depend on the API or frontend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


# ============================================================
# QUERY EVENT
# ============================================================

@dataclass
class QueryEvent:
    """
    Structured telemetry for a single Text-to-SQL request.
    """

    # --------------------------------------------------------
    # Identity / timing
    # --------------------------------------------------------

    request_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    timestamp: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    question: str = ""

    status: str = "UNKNOWN"

    latency_ms: int = 0

    # --------------------------------------------------------
    # SQL
    # --------------------------------------------------------

    sql_generated: bool = False

    sql_safe: bool = False

    sql_correction_attempted: bool = False

    sql_correction_count: int = 0

    sql_correction_applied: bool = False

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    cache_hit: bool = False

    # --------------------------------------------------------
    # Resource guard
    # --------------------------------------------------------

    resource_decision: str = ""

    # --------------------------------------------------------
    # Semantic evaluation
    # --------------------------------------------------------

    semantic_correct: bool | None = None

    semantic_score: float = 0.0

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_score: float = 0.0

    confidence_level: str = ""

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    tables_used: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error: str = ""

    # --------------------------------------------------------
    # Additional metadata
    # --------------------------------------------------------

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ========================================================
    # SERIALIZATION
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the event into a JSON-friendly dictionary.
        """

        return asdict(self)

    # ============================================================
    # OBSERVABILITY STORE
    # ============================================================


class ObservabilityStore:
    """
    In-memory store for query observability events.

    The store is intentionally lightweight for the first
    implementation. Persistence can be added later without
    changing QueryEvent itself.
    """

    def __init__(
        self,
        max_events: int = 1000,
    ) -> None:
        self.max_events = max_events
        self._events: list[QueryEvent] = []

    # ========================================================
    # RECORD EVENT
    # ========================================================

    def record(
        self,
        event: QueryEvent,
    ) -> None:
        """
        Record one query event.

        The store keeps only the most recent `max_events`
        entries.
        """

        self._events.append(event)

        if len(self._events) > self.max_events:
            self._events = self._events[
                -self.max_events:
            ]

    # ========================================================
    # GET EVENTS
    # ========================================================

    def get_events(
        self,
    ) -> list[QueryEvent]:
        """
        Return stored events in chronological order.
        """

        return list(self._events)

    # ========================================================
    # RECENT EVENTS
    # ========================================================

    def get_recent(
        self,
        limit: int = 20,
    ) -> list[QueryEvent]:
        """
        Return the most recent events.
        """

        if limit <= 0:
            return []

        return self._events[-limit:]

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """
        Remove all stored events.
        """

        self._events.clear()

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        """
        Return the number of stored events.
        """

        return len(self._events)

    # ============================================================
    # OBSERVABILITY METRICS
    # ============================================================


def calculate_metrics(
    events: list[QueryEvent],
) -> dict[str, Any]:
    """
    Calculate aggregate observability metrics from query events.

    Returns a JSON-friendly dictionary suitable for API
    responses and dashboard consumption.
    """

    total_queries = len(events)

    if total_queries == 0:
        return {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "success_rate": 0.0,
            "average_latency_ms": 0.0,
            "average_confidence": 0.0,
            "cache_hit_rate": 0.0,
            "sql_correction_rate": 0.0,
            "semantic_accuracy": 0.0,
            "safety_blocks": 0,
        }

    successful_queries = sum(
        1
        for event in events
        if event.status == "SUCCESS"
    )

    failed_queries = sum(
        1
        for event in events
        if event.status == "FAILED"
    )

    total_latency = sum(
        event.latency_ms
        for event in events
    )

    total_confidence = sum(
        event.confidence_score
        for event in events
    )

    cache_hits = sum(
        1
        for event in events
        if event.cache_hit
    )

    correction_attempts = sum(
        1
        for event in events
        if event.sql_correction_attempted
    )

    semantic_events = [
        event
        for event in events
        if event.semantic_correct is not None
    ]

    semantic_correct = sum(
        1
        for event in semantic_events
        if event.semantic_correct is True
    )

    safety_blocks = sum(
        1
        for event in events
        if event.resource_decision == "BLOCK"
    )

    return {
        "total_queries": total_queries,

        "successful_queries": successful_queries,

        "failed_queries": failed_queries,

        "success_rate": round(
            (
                successful_queries
                / total_queries
            )
            * 100,
            2,
        ),

        "average_latency_ms": round(
            total_latency
            / total_queries,
            2,
        ),

        "average_confidence": round(
            total_confidence
            / total_queries,
            2,
        ),

        "cache_hit_rate": round(
            (
                cache_hits
                / total_queries
            )
            * 100,
            2,
        ),

        "sql_correction_rate": round(
            (
                correction_attempts
                / total_queries
            )
            * 100,
            2,
        ),

        "semantic_accuracy": round(
            (
                semantic_correct
                / len(semantic_events)
            )
            * 100,
            2,
        )
        if semantic_events
        else 0.0,

        "safety_blocks": safety_blocks,
    }