import asyncio
from unittest.mock import AsyncMock

from agent.query_explainer import explain_query
import pytest

# ============================================================
# MOCK LLM
# ============================================================


def create_mock_llm(response_text: str):
    """Create a mock LLM returning a controlled explanation."""

    mock_llm = AsyncMock()

    mock_response = type(
        "MockResponse",
        (),
        {
            "content": response_text,
        },
    )()

    mock_llm.ainvoke.return_value = mock_response

    return mock_llm


# ============================================================
# TEST 1 — Revenue query
# ============================================================

@pytest.mark.asyncio
async def test_revenue_explanation():

    sql = """
    SELECT dp.category_name,
           ROUND(SUM(fo.order_total_usd), 2) AS total_revenue
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    WHERE fo.order_status = 'delivered'
    GROUP BY dp.category_name
    ORDER BY total_revenue DESC
    """

    mock_llm = create_mock_llm(
        "The query calculates total revenue for each product category using delivered orders, "
        "then sorts the categories from highest to lowest revenue."
    )

    explanation = await explain_query(
        question=(
            "Which product categories generated "
            "the highest total revenue?"
        ),
        sql=sql,
        tables_used=[
            "fact_orders",
            "dim_products",
        ],
        llm=mock_llm,
    )

    assert explanation["summary"]

    print("Summary generation: PASS")

    assert (
        "total revenue"
        in explanation["summary"].lower()
    )

    print("Revenue explanation: PASS")

    assert explanation["tables_used"] == [
        "fact_orders",
        "dim_products",
    ]

    print("Tables metadata: PASS")

    assert explanation["operation_count"] == 5

    print("Operation count: PASS")

    assert mock_llm.ainvoke.called

    print("LLM explanation call: PASS")

    print()
    print("Generated explanation:")
    print(explanation["summary"])


# ============================================================
# TEST 2 — COUNT DISTINCT delivered orders
# ============================================================

@pytest.mark.asyncio
async def test_distinct_delivered_orders():

    sql = """
    SELECT COUNT(DISTINCT fo.order_id) AS delivered_orders_count
    FROM fact_orders fo
    WHERE fo.order_status = 'delivered';
    """

    mock_llm = create_mock_llm(
        "The query counts distinct orders where the order status is delivered."
    )

    explanation = await explain_query(
        question="How many orders were delivered?",
        sql=sql,
        tables_used=[
            "fact_orders",
        ],
        llm=mock_llm,
    )

    assert (
        "counts distinct orders"
        in explanation["summary"].lower()
    )

    print("COUNT DISTINCT explanation: PASS")

    assert (
        "delivered"
        in explanation["summary"].lower()
    )

    print("Delivered filter explanation: PASS")

    assert explanation["tables_used"] == [
        "fact_orders",
    ]

    print("Tables metadata: PASS")

    assert explanation["operation_count"] == 2

    print("Operation count: PASS")


# ============================================================
# TEST 3 — Top 5 categories by order count
# ============================================================

@pytest.mark.asyncio
async def test_top_5_categories():

    sql = """
    SELECT dp.category_name,
           COUNT(DISTINCT fo.order_id) AS order_count
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    GROUP BY dp.category_name
    ORDER BY order_count DESC
    LIMIT 5
    """

    mock_llm = create_mock_llm(
        "The query counts distinct orders for each product category, "
        "ranks the categories by order count from highest to lowest, "
        "and returns the top 5 categories."
    )

    explanation = await explain_query(
        question=(
            "What are the top 5 product categories "
            "by number of orders?"
        ),
        sql=sql,
        tables_used=[
            "fact_orders",
            "dim_products",
        ],
        llm=mock_llm,
    )

    summary = explanation["summary"].lower()

    assert "order" in summary

    print("Order count explanation: PASS")

    assert "top 5" in summary

    print("LIMIT explanation: PASS")

    assert explanation["operation_count"] == 5

    print("Operation count: PASS")


# ============================================================
# TEST 4 — LLM failure fallback
# ============================================================

@pytest.mark.asyncio
async def test_llm_fallback():

    sql = """
    SELECT COUNT(DISTINCT fo.order_id) AS delivered_orders_count
    FROM fact_orders fo
    WHERE fo.order_status = 'delivered';
    """

    mock_llm = AsyncMock()

    mock_llm.ainvoke.side_effect = Exception(
        "Simulated LLM failure"
    )

    explanation = await explain_query(
        question="How many orders were delivered?",
        sql=sql,
        tables_used=[
            "fact_orders",
        ],
        llm=mock_llm,
    )

    assert explanation["summary"]

    print("LLM fallback summary: PASS")

    assert (
        "distinct orders"
        in explanation["summary"].lower()
    )

    print("Fallback grounding: PASS")

    assert explanation["tables_used"] == [
        "fact_orders",
    ]

    print("Fallback metadata: PASS")


# ============================================================
# TEST 5 — Adversarial grounding
# ============================================================

@pytest.mark.asyncio
async def test_grounding_rejects_wrong_revenue_claim():
    """
    Verify that the system rejects an LLM explanation
    that claims revenue when the SQL only counts orders.
    """

    sql = """
    SELECT COUNT(DISTINCT fo.order_id) AS order_count
    FROM fact_orders fo
    WHERE fo.order_status = 'canceled';
    """

    class BadLLM:

        async def ainvoke(self, prompt):

            class Response:
                content = (
                    "The query calculates total revenue "
                    "for delivered orders."
                )

            return Response()

    explanation = await explain_query(
        question="How many orders were cancelled?",
        sql=sql,
        tables_used=[
            "fact_orders",
        ],
        llm=BadLLM(),
    )

    summary = explanation["summary"].lower()

    assert "revenue" not in summary
    assert "delivered" not in summary

    print(
        "Adversarial revenue/status grounding: PASS"
    )

@pytest.mark.asyncio
async def test_grounding_rejects_wrong_order_claim():
    """
    Verify that the system rejects an LLM explanation
    that claims order counting when the SQL calculates revenue.
    """

    sql = """
    SELECT ROUND(
        SUM(fo.order_total_usd),
        2
    ) AS total_revenue
    FROM fact_orders fo;
    """

    class BadLLM:

        async def ainvoke(self, prompt):

            class Response:
                content = (
                    "The query counts the number "
                    "of unique orders."
                )

            return Response()

    explanation = await explain_query(
        question="What is the total revenue?",
        sql=sql,
        tables_used=[
            "fact_orders",
        ],
        llm=BadLLM(),
    )

    summary = explanation["summary"].lower()

    assert (
        "counts the number of unique orders"
        not in summary
    )

    print(
        "Adversarial order-count grounding: PASS"
    )


# ============================================================
# MAIN
# ============================================================


async def main():

    print("=" * 70)
    print("QUERY EXPLAINER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # TEST 1
    # --------------------------------------------------------

    print()
    print("TEST 1 — Revenue explanation")
    print("-" * 50)

    await test_revenue_explanation()

    # --------------------------------------------------------
    # TEST 2
    # --------------------------------------------------------

    print()
    print("TEST 2 — COUNT DISTINCT delivered orders")
    print("-" * 50)

    await test_distinct_delivered_orders()

    # --------------------------------------------------------
    # TEST 3
    # --------------------------------------------------------

    print()
    print("TEST 3 — Top 5 categories")
    print("-" * 50)

    await test_top_5_categories()

    # --------------------------------------------------------
    # TEST 4
    # --------------------------------------------------------

    print()
    print("TEST 4 — LLM failure fallback")
    print("-" * 50)

    await test_llm_fallback()

    # --------------------------------------------------------
    # TEST 5
    # --------------------------------------------------------

    print()
    print("TEST 5 — Adversarial grounding")
    print("-" * 50)

    await test_grounding_rejects_wrong_revenue_claim()
    await test_grounding_rejects_wrong_order_claim()

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL QUERY EXPLAINER TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())