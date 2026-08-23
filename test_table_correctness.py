"""
test_table_correctness.py

Tests for deterministic table correctness verification.
"""

from agent.table_correctness import (
    check_table_correctness,
    extract_sql_tables,
    extract_table_aliases,
    extract_qualified_columns,
)


# ============================================================
# TEST 1 — TABLE EXTRACTION
# ============================================================

def test_table_extraction():

    sql = """
    SELECT dp.category_name,
           SUM(fo.order_total_usd) AS revenue
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    GROUP BY dp.category_name
    """

    tables = extract_sql_tables(
        sql
    )

    assert tables == [
        "fact_orders",
        "dim_products",
    ]

    print(
        "Table extraction: PASS"
    )


# ============================================================
# TEST 2 — ALIAS EXTRACTION
# ============================================================

def test_alias_extraction():

    sql = """
    SELECT dp.category_name
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    """

    aliases = extract_table_aliases(
        sql
    )

    assert aliases["fo"] == (
        "fact_orders"
    )

    assert aliases["dp"] == (
        "dim_products"
    )

    print(
        "Alias extraction: PASS"
    )


# ============================================================
# TEST 3 — QUALIFIED COLUMN EXTRACTION
# ============================================================

def test_column_extraction():

    sql = """
    SELECT dp.category_name,
           SUM(fo.order_total_usd)
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    """

    columns = extract_qualified_columns(
        sql
    )

    assert (
        "dp",
        "category_name",
    ) in columns

    assert (
        "fo",
        "order_total_usd",
    ) in columns

    assert (
        "fo",
        "product_id",
    ) in columns

    assert (
        "dp",
        "product_id",
    ) in columns

    print(
        "Qualified column extraction: PASS"
    )


# ============================================================
# TEST 4 — VALID REVENUE QUERY
# ============================================================

def test_valid_revenue_query():

    sql = """
    SELECT dp.category_name,
           ROUND(
               SUM(fo.order_total_usd),
               2
           ) AS total_revenue
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    WHERE fo.order_status = 'delivered'
    GROUP BY dp.category_name
    ORDER BY total_revenue DESC
    """

    result = check_table_correctness(
        sql=sql
    )

    assert result["table_correct"] is True

    assert result["tables_used"] == [
        "fact_orders",
        "dim_products",
    ]

    assert result["invalid_tables"] == []

    assert result["invalid_columns"] == []

    print(
        "Valid revenue table correctness: PASS"
    )


# ============================================================
# TEST 5 — VALID CANCELLED ORDERS QUERY
# ============================================================

def test_valid_cancelled_orders_query():

    sql = """
    SELECT COUNT(DISTINCT fo.order_id)
    FROM fact_orders fo
    WHERE fo.order_status = 'canceled'
    """

    result = check_table_correctness(
        sql=sql
    )

    assert result["table_correct"] is True

    assert result["tables_used"] == [
        "fact_orders"
    ]

    print(
        "Valid cancelled orders table correctness: PASS"
    )


# ============================================================
# TEST 6 — INVALID TABLE
# ============================================================

def test_invalid_table():

    sql = """
    SELECT *
    FROM fake_orders fo
    """

    result = check_table_correctness(
        sql=sql
    )

    assert result["table_correct"] is False

    assert (
        "fake_orders"
        in result["invalid_tables"]
    )

    assert (
        "INVALID_TABLE_REFERENCE"
        in result["issues"]
    )

    print(
        "Invalid table detection: PASS"
    )


# ============================================================
# TEST 7 — INVALID COLUMN
# ============================================================

def test_invalid_column():

    sql = """
    SELECT
        fo.fake_revenue
    FROM fact_orders fo
    """

    result = check_table_correctness(
        sql=sql
    )

    assert result["table_correct"] is False

    assert (
        "fact_orders.fake_revenue"
        in result["invalid_columns"]
    )

    assert (
        "INVALID_COLUMN_REFERENCE"
        in result["issues"]
    )

    print(
        "Invalid column detection: PASS"
    )


# ============================================================
# TEST 8 — INVALID JOIN COLUMN
# ============================================================

def test_invalid_join_column():

    sql = """
    SELECT dp.category_name
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.fake_product_id = dp.product_id
    """

    result = check_table_correctness(
        sql=sql
    )

    assert result["table_correct"] is False

    assert (
        "fact_orders.fake_product_id"
        in result["invalid_columns"]
    )

    print(
        "Invalid join column detection: PASS"
    )


# ============================================================
# TEST 9 — EMPTY SQL
# ============================================================

def test_empty_sql():

    result = check_table_correctness(
        sql=""
    )

    assert result["table_correct"] is False

    assert (
        "EMPTY_SQL"
        in result["issues"]
    )

    print(
        "Empty SQL validation: PASS"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "TABLE CORRECTNESS TEST SUITE"
    )
    print("=" * 70)

    test_table_extraction()
    test_alias_extraction()
    test_column_extraction()
    test_valid_revenue_query()
    test_valid_cancelled_orders_query()
    test_invalid_table()
    test_invalid_column()
    test_invalid_join_column()
    test_empty_sql()

    print()
    print("=" * 70)
    print(
        "ALL TABLE CORRECTNESS TESTS PASSED"
    )
    print("=" * 70)