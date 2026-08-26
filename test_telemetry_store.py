from pathlib import Path

from agent.observability import QueryEvent
from agent.telemetry_store import (
    get_recent_events,
    get_summary,
    record_event,
)


def test_record_and_retrieve_event(
    tmp_path: Path,
):

    db_path = tmp_path / "telemetry.db"

    event = QueryEvent(
        question="What is the total revenue?",
        status="SUCCESS",
        latency_ms=1200,
        sql_generated=True,
        sql_safe=True,
        cache_hit=False,
        resource_decision="ALLOW",
        semantic_correct=True,
        semantic_score=0.95,
        confidence_score=98.5,
        confidence_level="HIGH",
        tables_used=["fact_orders"],
    )

    record_event(
        event,
        db_path,
    )

    events = get_recent_events(
        db_path=db_path,
    )

    assert len(events) == 1

    assert events[0]["request_id"] == event.request_id

    assert events[0]["question"] == (
        "What is the total revenue?"
    )

    assert events[0]["status"] == "SUCCESS"

    assert events[0]["sql_generated"] is True

    assert events[0]["semantic_correct"] is True

    assert events[0]["tables_used"] == [
        "fact_orders"
    ]


def test_summary_metrics(
    tmp_path: Path,
):

    db_path = tmp_path / "telemetry.db"

    record_event(
        QueryEvent(
            question="Revenue",
            status="SUCCESS",
            latency_ms=1000,
            sql_generated=True,
            sql_safe=True,
            cache_hit=True,
            semantic_correct=True,
            semantic_score=1.0,
            confidence_score=100,
            confidence_level="HIGH",
        ),
        db_path,
    )

    record_event(
        QueryEvent(
            question="Orders",
            status="FAILED",
            latency_ms=3000,
            sql_generated=True,
            sql_safe=True,
            cache_hit=False,
            semantic_correct=False,
            semantic_score=0.2,
            confidence_score=40,
            confidence_level="LOW",
        ),
        db_path,
    )

    summary = get_summary(
        db_path=db_path,
    )

    assert summary["total_queries"] == 2

    assert summary["successful_queries"] == 1

    assert summary["failed_queries"] == 1

    assert summary["success_rate"] == 0.5

    assert summary["cache_hits"] == 1

    assert summary["cache_hit_rate"] == 0.5

    assert summary["avg_latency_ms"] == 2000.0

    assert summary["avg_confidence_score"] == 70.0