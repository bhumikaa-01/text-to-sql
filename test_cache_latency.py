import asyncio

from agent.query_cache import (
    clear_cache,
    set_cached_response,
)
from agent.sql_chain import run_query

from unittest.mock import patch
import pytest

@pytest.mark.asyncio
async def test_cache_hit_latency():

    print("=" * 70)
    print("CACHE HIT LATENCY TEST")
    print("=" * 70)

    clear_cache()

    question = "What is the average order value for delivered orders?"

    cached_response = {
        "sql": "SELECT AVG(order_total_usd) FROM fact_orders;",
        "results": [
            {"average_order_value": 123.45}
        ],
        "tables_used": ["fact_orders"],
        "requires_approval": False,
        "approval_reason": "",
        "resource_guard": {
            "decision": "ALLOW",
            "risk_level": "LOW",
            "violations": [],
            "reason": "",
        },
        "semantic_evaluation": {
            "is_correct": True,
            "score": 1.0,
            "reason": "Cached test response",
            "issues": [],
        },
        "confidence": {
            "score": 90,
            "level": "HIGH",
        },
        "latency_ms": 6500,
        "error": "",
    }

    # Pre-populate cache.
    set_cached_response(
        question,
        cached_response,
    )

    print("\nCached response created: PASS")

    # Run the REAL pipeline.
    # Because the question is cached, Gemini should NOT be called.
    result = await run_query(question)

    print("\nCache:", result["cache"])
    print("Latency:", result["latency_ms"], "ms")

    assert result["cache"]["hit"] is True
    print("Cache HIT: PASS")

    assert result["latency_ms"] < 1000
    print("Cache latency < 1 second: PASS")

    print("\nOriginal cached latency:", cached_response["latency_ms"], "ms")
    print("Actual HIT latency:", result["latency_ms"], "ms")

    print("\n" + "=" * 70)
    print("CACHE HIT LATENCY TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(
        test_cache_hit_latency()
    )