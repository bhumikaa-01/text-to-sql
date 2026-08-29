import asyncio
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from agent.sql_chain import run_query
import pytest

@pytest.mark.asyncio
async def test_sql_retry_integration():

    print("=" * 70)
    print("SQL AUTOMATIC CORRECTION INTEGRATION TEST")
    print("=" * 70)

    responses = [
    # 1. Initial SQL generation
    """
    SELECT SUM(revenue)
    FROM fact_orders;
    """,

    # 2. SQL correction
    """
    SELECT SUM(order_total_usd) AS total_revenue
    FROM fact_orders;
    """,

    # 3. Query explanation
    """
    The query calculates the total revenue by summing
    order_total_usd across all orders.
    """,
]

    state = {
        "calls": 0,
    }

    async def fake_llm(_input):

        index = state["calls"]

        state["calls"] += 1

        return responses[index]

    mock_llm = RunnableLambda(
        fake_llm
    )

    with patch(
        "agent.sql_chain._get_llm",
        return_value=mock_llm,
    ), patch(
        "agent.sql_chain.get_cached_response",
        return_value=None,
    ), patch(
        "agent.sql_chain.set_cached_response",
    ):

        result = await run_query(
            "What is the total revenue?"
        )

    print("\nFinal SQL:")
    print(result["sql"])

    print("\nResults:")
    print(result["results"])

    print(
        "\nLLM calls:",
        state["calls"],
    )

    assert state["calls"] == 3

    print(
        "Automatic correction triggered: PASS"
    )

    assert (
        "order_total_usd"
        in result["sql"]
    )

    print(
        "Corrected SQL generated: PASS"
    )

    assert result["error"] == ""

    print(
        "Query completed successfully: PASS"
    )

    assert result["results"]

    print(
        "Database execution after correction: PASS"
    )

    print("\n" + "=" * 70)
    print(
        "SQL AUTOMATIC CORRECTION "
        "INTEGRATION TEST PASSED"
    )
    print("=" * 70)

@pytest.mark.asyncio
async def test_sql_retry_max_retries():

    print("=" * 70)
    print("SQL MAX RETRY LIMIT INTEGRATION TEST")
    print("=" * 70)

    responses = [
    # 1. Initial SQL generation
    """
    SELECT SUM(order_total_usd)
    FROM fact_orders;
    """,

    # 2. Correction #1
    """
    SELECT SUM(order_total_usd)
    FROM fact_orders
    WHERE order_status = 'delivered';
    """,

    # 3. Correction #2
    """
    SELECT SUM(order_total_usd)
    FROM fact_orders
    WHERE order_status = 'canceled';
    """,
    ]

    state = {
        "calls": 0,
    }

    async def fake_llm(_input):

        index = state["calls"]

        state["calls"] += 1

        return responses[index]

    mock_llm = RunnableLambda(
        fake_llm
    )

    def fake_execute_sql(_sql):

        raise Exception(
            "syntax error: simulated database failure"
        )

    with patch(
        "agent.sql_chain._get_llm",
        return_value=mock_llm,
    ), patch(
        "agent.sql_chain.get_cached_response",
        return_value=None,
    ), patch(
        "agent.sql_chain.set_cached_response",
    ), patch(
        "agent.sql_chain._execute_sql",
        side_effect=fake_execute_sql,
    ):

        result = await run_query(
            "What is the total revenue?"
        )

    print("\nLLM calls:")
    print(state["calls"])

    print("\nFinal SQL:")
    print(result["sql"])

    print("\nError:")
    print(result["error"])

    print("\nRetry metadata:")
    print(result["sql_retry"])

    # --------------------------------------------------------
    # Verify exactly two correction attempts occurred.
    # --------------------------------------------------------

    assert state["calls"] == 3

    print(
        "Initial generation + 2 corrections: PASS"
    )

    # --------------------------------------------------------
    # Verify retry metadata.
    # --------------------------------------------------------

    assert result["sql_retry"]["attempted"] is True

    assert (
        result["sql_retry"]["retry_count"] == 2
    )

    print(
        "MAX_SQL_RETRIES respected: PASS"
    )

    # --------------------------------------------------------
    # Verify no third correction was attempted.
    # --------------------------------------------------------

    assert (
        result["sql_retry"]["max_retries"] == 2
    )

    print(
        "No correction beyond retry limit: PASS"
    )

    # --------------------------------------------------------
    # Verify controlled failure.
    # --------------------------------------------------------

    assert result["results"] == []

    assert result["error"] != ""

    print(
        "Controlled failure returned: PASS"
    )

    print("\n" + "=" * 70)
    print(
        "SQL MAX RETRY LIMIT "
        "INTEGRATION TEST PASSED"
    )
    print("=" * 70)

if __name__ == "__main__":

    asyncio.run(
        test_sql_retry_integration()
    )