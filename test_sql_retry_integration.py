import asyncio
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from agent.sql_chain import run_query


async def test_sql_retry_integration():

    print("=" * 70)
    print("SQL AUTOMATIC CORRECTION INTEGRATION TEST")
    print("=" * 70)

    responses = [
        """
        SELECT SUM(revenue)
        FROM fact_orders;
        """,
        """
        SELECT SUM(order_total_usd) AS total_revenue
        FROM fact_orders;
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

    assert state["calls"] == 2

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


if __name__ == "__main__":

    asyncio.run(
        test_sql_retry_integration()
    )