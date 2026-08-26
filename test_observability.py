from datetime import datetime

from agent.observability import QueryEvent


def test_query_event_defaults():
    event = QueryEvent()

    assert event.request_id
    assert event.timestamp
    assert event.status == "UNKNOWN"
    assert event.latency_ms == 0

    assert event.sql_generated is False
    assert event.sql_safe is False

    assert event.sql_correction_attempted is False
    assert event.sql_correction_count == 0
    assert event.sql_correction_applied is False

    assert event.cache_hit is False

    assert event.semantic_correct is None
    assert event.semantic_score == 0.0

    assert event.confidence_score == 0.0
    assert event.confidence_level == ""

    assert event.tables_used == []
    assert event.error == ""
    assert event.metadata == {}


def test_query_event_has_unique_request_ids():
    event_one = QueryEvent()
    event_two = QueryEvent()

    assert event_one.request_id != event_two.request_id


def test_query_event_timestamp_is_valid_iso_format():
    event = QueryEvent()

    parsed = datetime.fromisoformat(
        event.timestamp
    )

    assert parsed.tzinfo is not None


def test_query_event_serialization():
    event = QueryEvent(
        question="What is the total revenue?",
        status="SUCCESS",
        latency_ms=1250,
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

    data = event.to_dict()

    assert data["question"] == (
        "What is the total revenue?"
    )

    assert data["status"] == "SUCCESS"
    assert data["latency_ms"] == 1250

    assert data["sql_generated"] is True
    assert data["sql_safe"] is True

    assert data["cache_hit"] is False

    assert data["resource_decision"] == "ALLOW"

    assert data["semantic_correct"] is True
    assert data["semantic_score"] == 0.95

    assert data["confidence_score"] == 98.5
    assert data["confidence_level"] == "HIGH"

    assert data["tables_used"] == [
        "fact_orders"
    ]


def test_query_event_tracks_sql_correction():
    event = QueryEvent(
        status="SUCCESS",
        sql_generated=True,
        sql_safe=True,
        sql_correction_attempted=True,
        sql_correction_count=1,
        sql_correction_applied=True,
    )

    data = event.to_dict()

    assert data["sql_correction_attempted"] is True
    assert data["sql_correction_count"] == 1
    assert data["sql_correction_applied"] is True


def test_query_event_tracks_failure():
    event = QueryEvent(
        question="Show invalid data",
        status="FAILED",
        latency_ms=500,
        sql_generated=True,
        sql_safe=True,
        error="SQL execution failed",
    )

    data = event.to_dict()

    assert data["status"] == "FAILED"
    assert data["error"] == "SQL execution failed"
    assert data["latency_ms"] == 500


def test_query_event_metadata():
    event = QueryEvent(
        metadata={
            "model": "gemini-2.5-flash-lite",
            "environment": "development",
        }
    )

    data = event.to_dict()

    assert data["metadata"]["model"] == (
        "gemini-2.5-flash-lite"
    )

    assert data["metadata"]["environment"] == (
        "development"
    )

def test_observability_store_records_events():
    from agent.observability import ObservabilityStore

    store = ObservabilityStore()

    event = QueryEvent(
        question="What is the total revenue?",
        status="SUCCESS",
    )

    store.record(event)

    assert store.count() == 1
    assert store.get_events()[0] is event


def test_observability_store_recent_events():
    from agent.observability import ObservabilityStore

    store = ObservabilityStore()

    first = QueryEvent(
        question="Question 1",
    )

    second = QueryEvent(
        question="Question 2",
    )

    third = QueryEvent(
        question="Question 3",
    )

    store.record(first)
    store.record(second)
    store.record(third)

    recent = store.get_recent(2)

    assert len(recent) == 2
    assert recent[0] is second
    assert recent[1] is third


def test_observability_store_max_events():
    from agent.observability import ObservabilityStore

    store = ObservabilityStore(
        max_events=2,
    )

    first = QueryEvent(
        question="Question 1",
    )

    second = QueryEvent(
        question="Question 2",
    )

    third = QueryEvent(
        question="Question 3",
    )

    store.record(first)
    store.record(second)
    store.record(third)

    events = store.get_events()

    assert len(events) == 2
    assert events[0] is second
    assert events[1] is third


def test_observability_store_clear():
    from agent.observability import ObservabilityStore

    store = ObservabilityStore()

    store.record(
        QueryEvent(
            question="Test",
        )
    )

    assert store.count() == 1

    store.clear()

    assert store.count() == 0
    assert store.get_events() == []

def test_calculate_metrics_empty_events():
    from agent.observability import calculate_metrics

    metrics = calculate_metrics([])

    assert metrics["total_queries"] == 0
    assert metrics["successful_queries"] == 0
    assert metrics["failed_queries"] == 0
    assert metrics["success_rate"] == 0.0
    assert metrics["average_latency_ms"] == 0.0
    assert metrics["average_confidence"] == 0.0
    assert metrics["cache_hit_rate"] == 0.0
    assert metrics["sql_correction_rate"] == 0.0
    assert metrics["semantic_accuracy"] == 0.0
    assert metrics["safety_blocks"] == 0


def test_calculate_metrics():
    from agent.observability import calculate_metrics

    events = [
        QueryEvent(
            status="SUCCESS",
            latency_ms=1000,
            confidence_score=90.0,
            cache_hit=False,
            sql_correction_attempted=False,
            semantic_correct=True,
            resource_decision="ALLOW",
        ),
        QueryEvent(
            status="SUCCESS",
            latency_ms=3000,
            confidence_score=80.0,
            cache_hit=True,
            sql_correction_attempted=True,
            semantic_correct=True,
            resource_decision="ALLOW",
        ),
        QueryEvent(
            status="FAILED",
            latency_ms=2000,
            confidence_score=50.0,
            cache_hit=False,
            sql_correction_attempted=True,
            semantic_correct=False,
            resource_decision="BLOCK",
        ),
    ]

    metrics = calculate_metrics(events)

    assert metrics["total_queries"] == 3
    assert metrics["successful_queries"] == 2
    assert metrics["failed_queries"] == 1

    assert metrics["success_rate"] == 66.67

    assert metrics["average_latency_ms"] == 2000.0

    assert metrics["average_confidence"] == 73.33

    assert metrics["cache_hit_rate"] == 33.33

    assert metrics["sql_correction_rate"] == 66.67

    assert metrics["semantic_accuracy"] == 66.67

    assert metrics["safety_blocks"] == 1