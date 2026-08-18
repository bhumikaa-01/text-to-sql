import time

from agent.query_cache import (
    clear_cache,
    get_cached_response,
    set_cached_response,
)


def main() -> None:

    print("=" * 70)
    print("QUERY CACHE TTL TEST")
    print("=" * 70)

    clear_cache()

    question = "What is the total revenue?"

    response = {
        "sql": "SELECT SUM(order_total_usd) FROM fact_orders;",
        "results": [
            {"total_revenue": 1000}
        ],
        "confidence": {
            "score": 90,
            "level": "HIGH",
        },
    }

    # --------------------------------------------------------
    # Store cache entry
    # --------------------------------------------------------

    set_cached_response(
        question,
        response,
    )

    print("\nCACHE CREATED")
    print("PASS")

    # --------------------------------------------------------
    # Verify cache HIT
    # --------------------------------------------------------

    cached = get_cached_response(
        question,
        ttl_seconds=2,
    )

    print("\nCACHE BEFORE TTL")
    print(cached)

    assert cached == response

    # --------------------------------------------------------
    # Wait until TTL expires
    # --------------------------------------------------------

    print("\nWAITING FOR TTL EXPIRY...")

    time.sleep(3)

    # --------------------------------------------------------
    # Verify cache MISS after expiry
    # --------------------------------------------------------

    cached = get_cached_response(
        question,
        ttl_seconds=2,
    )

    print("\nCACHE AFTER TTL")
    print(cached)

    assert cached is None

    print("\n" + "=" * 70)
    print("QUERY CACHE TTL TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()