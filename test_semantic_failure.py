"""
test_semantic_failure.py

Integration test for graceful semantic-evaluation failure.

No real Gemini API calls are made.

We mock:
    1. SQL-generation LLM using a LangChain Runnable
    2. Semantic evaluator failure

The real database execution remains active.
"""

import asyncio
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from agent.sql_chain import run_query
from agent.query_cache import clear_cache


async def test_semantic_evaluation_failure():

    print()
    print("=" * 70)
    print("SEMANTIC EVALUATION FAILURE TEST")
    print("=" * 70)

    mocked_sql = """
        SELECT
            ROUND(
                SUM(order_total_usd),
                2
            ) AS total_revenue
        FROM fact_orders
        WHERE order_status = 'delivered';
    """

    # --------------------------------------------------------
    # Mock LLM as a real LangChain Runnable.
    #
    # This prevents ANY Gemini request while still allowing
    # the real LCEL chain:
    #
    #     prompt | llm | StrOutputParser()
    #
    # to work normally.
    # --------------------------------------------------------

    mock_llm = RunnableLambda(
        lambda _: mocked_sql
    )

    # --------------------------------------------------------
    # Simulate semantic evaluator failure.
    # --------------------------------------------------------

    with patch(
        "agent.sql_chain._get_llm",
        return_value=mock_llm,
    ), patch(
        "agent.sql_chain.evaluate_semantics",
        side_effect=RuntimeError(
            "Simulated Gemini 503 Service Unavailable"
        ),
    ):

        # ----------------------------------------------------
        # Run the REAL query pipeline.
        # ----------------------------------------------------

        # Ensure the test exercises the real pipeline
        # instead of returning a previously cached response.
        clear_cache()

        result = await run_query(
            "What is the total revenue from delivered orders?"
        )

    # --------------------------------------------------------
    # Inspect result.
    # --------------------------------------------------------

    print(
        "Semantic evaluation:",
        result["semantic_evaluation"],
    )

    print(
        "Results returned:",
        result["results"],
    )

    print(
        "Error:",
        result["error"],
    )

    # --------------------------------------------------------
    # Semantic evaluator should be marked unavailable.
    # --------------------------------------------------------

    assert (
        result["semantic_evaluation"]["is_correct"]
        is None
    )

    assert (
        result["semantic_evaluation"]["score"]
        is None
    )

    assert (
        "SEMANTIC_EVALUATION_UNAVAILABLE"
        in result["semantic_evaluation"]["issues"]
    )

    # --------------------------------------------------------
    # Critical production guarantee:
    #
    # Semantic evaluation failure must NOT destroy
    # an otherwise successful database query.
    # --------------------------------------------------------

    assert result["results"]

    assert result["error"] == ""

    print()
    print(
        "Semantic failure handled: PASS"
    )

    print(
        "Database result preserved: PASS"
    )

    print()
    print("=" * 70)
    print(
        "SEMANTIC FAILURE TEST PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":

    asyncio.run(
        test_semantic_evaluation_failure()
    )