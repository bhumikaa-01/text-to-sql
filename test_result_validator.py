"""
test_result_validator.py

Tests for the Result / Semantic Validation Guard.
"""

from agent.result_validator import validate_query_result


# ============================================================
# PASS — VALID REVENUE
# ============================================================

def test_valid_revenue():

    result = validate_query_result(
        question="What is the total revenue?",
        generated_sql="""
            SELECT
                SUM(order_total_usd) AS total_revenue
            FROM fact_orders
        """,
        results=[
            {
                "total_revenue": 2644299.62
            }
        ],
    )

    print()
    print("=" * 70)
    print("VALID REVENUE")
    print("=" * 70)
    print(result)

    assert result["status"] == "PASS"
    assert result["score"] == 100.0


# ============================================================
# WARN — EMPTY RESULT
# ============================================================

def test_empty_result():

    result = validate_query_result(
        question="Show me delivered orders",
        generated_sql="""
            SELECT
                order_id
            FROM fact_orders
            WHERE order_status = 'delivered'
        """,
        results=[],
    )

    print()
    print("=" * 70)
    print("EMPTY RESULT")
    print("=" * 70)
    print(result)

    assert result["status"] == "WARN"
    assert result["score"] == 60.0


# ============================================================
# PASS — GROUPED RESULT
# ============================================================

def test_grouped_result():

    result = validate_query_result(
        question="Show revenue by month",
        generated_sql="""
            SELECT
                strftime(
                    '%Y-%m',
                    created_at
                ) AS month,
                SUM(order_total_usd) AS revenue
            FROM fact_orders
            GROUP BY strftime(
                '%Y-%m',
                created_at
            )
        """,
        results=[
            {
                "month": "2026-01",
                "revenue": 120000.0,
            },
            {
                "month": "2026-02",
                "revenue": 135000.0,
            },
        ],
    )

    print()
    print("=" * 70)
    print("GROUPED RESULT")
    print("=" * 70)
    print(result)

    assert result["status"] == "PASS"
    assert result["score"] == 100.0


# ============================================================
# WARN — NULL-HEAVY RESULT
# ============================================================

def test_null_heavy_result():

    result = validate_query_result(
        question="Show product categories",
        generated_sql="""
            SELECT
                category_name
            FROM dim_products
            LIMIT 10
        """,
        results=[
            {
                "category_name": None
            },
            {
                "category_name": None
            },
            {
                "category_name": None
            },
            {
                "category_name": "electronics"
            },
            {
                "category_name": None
            },
        ],
    )

    print()
    print("=" * 70)
    print("NULL-HEAVY RESULT")
    print("=" * 70)
    print(result)

    assert result["status"] == "WARN"
    assert result["score"] == 60.0


# ============================================================
# FAIL — WRONG RESULT SHAPE
# ============================================================

def test_wrong_result_shape():

    result = validate_query_result(
        question="What is the total revenue?",
        generated_sql="""
            SELECT
                SUM(order_total_usd)
            FROM fact_orders
        """,
        results={
            "total_revenue": 2644299.62
        },
    )

    print()
    print("=" * 70)
    print("WRONG RESULT SHAPE")
    print("=" * 70)
    print(result)

    assert result["status"] == "FAIL"
    assert result["score"] == 0.0


# ============================================================
# FAIL — SEMANTIC MISMATCH
# ============================================================

def test_semantic_mismatch():

    result = validate_query_result(
        question="What is the total revenue?",
        generated_sql="""
            SELECT
                AVG(freight_value_usd) AS average_shipping
            FROM fact_orders
        """,
        results=[
            {
                "average_shipping": 18.42
            }
        ],
    )

    print()
    print("=" * 70)
    print("SEMANTIC MISMATCH")
    print("=" * 70)
    print(result)

    assert result["status"] == "FAIL"
    assert result["score"] == 0.0


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    test_valid_revenue()

    test_empty_result()

    test_grouped_result()

    test_null_heavy_result()

    test_wrong_result_shape()

    test_semantic_mismatch()

    print()
    print("=" * 70)
    print("ALL RESULT VALIDATOR TESTS PASSED")
    print("=" * 70)