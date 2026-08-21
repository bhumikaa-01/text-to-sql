import asyncio

from agent.sql_chain import run_query


async def main():
    print("=" * 70)
    print("QUERY EXPLANATION INTEGRATION TEST")
    print("=" * 70)

    question = (
        "Which product categories generated the highest total revenue?"
    )

    result = await run_query(question)

    # --------------------------------------------------
    # Explanation exists
    # --------------------------------------------------

    assert "explanation" in result
    print("Explanation present: PASS")

    explanation = result["explanation"]

    # --------------------------------------------------
    # Explanation structure
    # --------------------------------------------------

    assert explanation["summary"]
    print("Explanation summary: PASS")

    assert explanation["tables_used"]
    print("Explanation tables metadata: PASS")

    assert explanation["operation_count"] > 0
    print("Explanation operation detection: PASS")

    # --------------------------------------------------
    # Expected operations
    # --------------------------------------------------

    summary = explanation["summary"]

    assert "calculates total revenue by summing `order_total_usd`" in summary
    print("SUM explanation: PASS")

    assert (
        "joining orders with product information "
        "to associate revenue with product categories"
        in summary
    )
    print("JOIN explanation: PASS")

    assert "using only delivered orders" in summary
    print("WHERE explanation: PASS")

    assert "grouping the results by product category" in summary
    print("GROUP BY explanation: PASS")

    assert "sorting the results from highest to lowest" in summary
    print("ORDER BY explanation: PASS")

    # --------------------------------------------------
    # Cache information
    # --------------------------------------------------

    assert "cache" in result
    print("Cache metadata: PASS")

    print()
    print("Generated explanation:")
    print(summary)

    print()
    print("=" * 70)
    print("QUERY EXPLANATION INTEGRATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())