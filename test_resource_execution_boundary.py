"""
test_resource_execution_boundary.py

Integration test for the real run_query() execution boundary.

Verifies that a BLOCK decision from the resource guard
prevents SQL from reaching the database executor.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from agent import sql_chain


async def test_block_does_not_reach_database():

    question = (
        "Show me all orders with a huge result set"
    )

    blocked_sql = """
    SELECT
        order_id
    FROM fact_orders
    LIMIT 500000
    """

    # --------------------------------------------------------
    # Mock the LLM chain
    # --------------------------------------------------------

    class FakeChain:

        async def ainvoke(
            self,
            inputs,
        ):
            return blocked_sql

        def __or__(
            self,
            other,
        ):
            return self

    # --------------------------------------------------------
    # Mock database execution
    # --------------------------------------------------------

    mock_execute = AsyncMock(
        return_value=[]
    )

    with patch.object(
        sql_chain,
        "get_relevant_schema",
        return_value="fact_orders schema",
    ), patch.object(
        sql_chain,
        "_load_few_shot_examples",
        return_value="",
    ), patch.object(
        sql_chain,
        "_get_llm",
        return_value=object(),
    ), patch.object(
        sql_chain.ChatPromptTemplate,
        "from_messages",
        return_value=FakeChain(),
    ), patch.object(
        sql_chain,
        "_execute_sql",
        mock_execute,
    ):

        # ----------------------------------------------------
        # Execute the REAL run_query()
        # ----------------------------------------------------

        result = await sql_chain.run_query(
            question
        )

    # ========================================================
    # Assertions
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK EXECUTION BOUNDARY TEST"
    )
    print("=" * 70)

    print(
        "Resource decision:",
        result["resource_guard"]["decision"],
    )

    print(
        "Risk level:",
        result["resource_guard"]["risk_level"],
    )

    print(
        "Violations:",
        result["resource_guard"]["violations"],
    )

    print(
        "Error:",
        result["error"],
    )

    # --------------------------------------------------------
    # Resource guard must BLOCK
    # --------------------------------------------------------

    assert (
        result["resource_guard"]["decision"]
        == "BLOCK"
    )

    assert (
        result["resource_guard"]["risk_level"]
        == "HIGH"
    )

    assert (
        "EXCESSIVE_LIMIT"
        in result["resource_guard"]["violations"]
    )

    # --------------------------------------------------------
    # No database execution
    # --------------------------------------------------------

    mock_execute.assert_not_awaited()

    # --------------------------------------------------------
    # No results should be returned
    # --------------------------------------------------------

    assert result["results"] == []

    print()
    print(
        "Resource guard blocked query: PASS"
    )

    print(
        "Database execution prevented: PASS"
    )

    print()
    print("=" * 70)
    print(
        "RESOURCE EXECUTION BOUNDARY TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    asyncio.run(
        test_block_does_not_reach_database()
    )