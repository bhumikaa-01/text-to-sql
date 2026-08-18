from agent.query_cache import (
    clear_cache,
    get_cached_response,
    set_cached_response,
)


def main() -> None:

    print("=" * 70)
    print("QUERY CACHE TEST")
    print("=" * 70)

    clear_cache()

    question = "  What is the TOTAL revenue?  "

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
    # CACHE MISS
    # --------------------------------------------------------

    cached = get_cached_response(question)

    print("\nCACHE MISS")
    print(cached)

    assert cached is None

    # --------------------------------------------------------
    # CACHE WRITE
    # --------------------------------------------------------

    set_cached_response(
        question,
        response,
    )

    print("\nCACHE WRITE")
    print("PASS")

    # --------------------------------------------------------
    # CACHE HIT
    # --------------------------------------------------------

    cached = get_cached_response(
        "what is the total revenue?"
    )

    print("\nCACHE HIT")
    print(cached)

    assert cached == response

    # --------------------------------------------------------
    # NORMALIZATION TEST
    # --------------------------------------------------------

    cached = get_cached_response(
        "   WHAT   IS   THE   TOTAL   REVENUE?   "
    )

    print("\nNORMALIZATION")
    print(cached)

    assert cached == response

    print("\n" + "=" * 70)
    print("ALL QUERY CACHE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()