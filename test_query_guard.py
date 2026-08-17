from agent.query_guard import (
    check_query_resources,
)


def test_safe_query():

    result = check_query_resources(
        """
        SELECT
            order_status,
            COUNT(*) AS order_count
        FROM fact_orders
        GROUP BY order_status
        """
    )

    assert result["allowed"] is True
    assert result["decision"] == "ALLOW"
    assert result["risk_level"] == "LOW"
    assert result["violations"] == []


def test_large_limit_warning():

    result = check_query_resources(
        """
        SELECT
            order_id
        FROM fact_orders
        LIMIT 800
        """
    )

    assert result["allowed"] is True
    assert result["decision"] == "WARN"
    assert result["risk_level"] == "MEDIUM"

    assert (
        "LARGE_LIMIT"
        in result["violations"]
    )


def test_excessive_limit_block():

    result = check_query_resources(
        """
        SELECT
            order_id
        FROM fact_orders
        LIMIT 500000
        """
    )

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"
    assert result["risk_level"] == "HIGH"

    assert (
        "EXCESSIVE_LIMIT"
        in result["violations"]
    )


def test_select_star_warning():

    result = check_query_resources(
        """
        SELECT *
        FROM fact_orders
        """
    )

    assert result["allowed"] is True
    assert result["decision"] == "WARN"
    assert result["risk_level"] == "MEDIUM"

    assert (
        "SELECT_STAR"
        in result["violations"]
    )


def test_select_star_with_limit_warning():

    result = check_query_resources(
        """
        SELECT *
        FROM fact_orders
        LIMIT 100
        """
    )

    assert result["allowed"] is True
    assert result["decision"] == "WARN"

    assert (
        "SELECT_STAR"
        in result["violations"]
    )


def test_complex_joins_warning():

    result = check_query_resources(
        """
        SELECT
            fo.order_id
        FROM fact_orders fo

        JOIN dim_users du
            ON fo.user_id = du.user_id

        JOIN dim_products dp
            ON fo.product_id = dp.product_id

        JOIN dim_sellers ds
            ON fo.seller_id = ds.seller_id

        JOIN dim_reviews dr
            ON fo.order_id = dr.order_id
        """
    )

    assert result["allowed"] is True
    assert result["decision"] == "WARN"
    assert result["risk_level"] == "MEDIUM"

    assert (
        "COMPLEX_JOINS"
        in result["violations"]
    )


def test_excessive_joins_block():

    result = check_query_resources(
        """
        SELECT *
        FROM fact_orders fo

        JOIN dim_users du
            ON fo.user_id = du.user_id

        JOIN dim_products dp
            ON fo.product_id = dp.product_id

        JOIN dim_sellers ds
            ON fo.seller_id = ds.seller_id

        JOIN dim_reviews dr
            ON fo.order_id = dr.order_id

        JOIN dim_geography dg
            ON du.state = dg.state

        JOIN another_table at
            ON at.id = fo.order_id
        """
    )

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"
    assert result["risk_level"] == "HIGH"

    assert (
        "EXCESSIVE_JOINS"
        in result["violations"]
    )


def test_complex_unions_warning():

    result = check_query_resources(
        """
        SELECT order_id FROM fact_orders
        UNION
        SELECT order_id FROM fact_orders
        UNION
        SELECT order_id FROM fact_orders
        UNION
        SELECT order_id FROM fact_orders
        """
    )

    assert result["allowed"] is True
    assert result["decision"] == "WARN"
    assert result["risk_level"] == "MEDIUM"

    assert (
        "COMPLEX_UNIONS"
        in result["violations"]
    )


def test_excessive_unions_block():

    result = check_query_resources(
        """
        SELECT order_id FROM fact_orders
        UNION
        SELECT order_id FROM fact_orders
        UNION
        SELECT order_id FROM fact_orders
        UNION
        SELECT order_id FROM fact_orders
        UNION
        SELECT order_id FROM fact_orders
        """
    )

    assert result["allowed"] is False
    assert result["decision"] == "BLOCK"
    assert result["risk_level"] == "HIGH"

    assert (
        "EXCESSIVE_UNIONS"
        in result["violations"]
    )

if __name__ == "__main__":

    test_safe_query()

    test_large_limit_warning()

    test_excessive_limit_block()

    test_select_star_warning()

    test_select_star_with_limit_warning()

    test_complex_joins_warning()

    test_excessive_joins_block()

    test_complex_unions_warning()

    test_excessive_unions_block()

    print(
        "All query resource guard tests passed."
    )