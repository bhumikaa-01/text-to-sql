"""
test_resource_pipeline.py

Integration tests for the query resource policy.

Verifies that:

    ALLOW
        → query is permitted

    WARN
        → query is permitted with warning metadata

    BLOCK
        → query is rejected before execution
"""

from agent.query_guard import check_query_resources


def print_result(
    name: str,
    result: dict,
) -> None:

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Allowed      :",
        result["allowed"],
    )

    print(
        "Decision     :",
        result["decision"],
    )

    print(
        "Risk level   :",
        result["risk_level"],
    )

    print(
        "Violations   :",
        result["violations"],
    )

    print(
        "Reason       :",
        result["reason"],
    )


# ============================================================
# ALLOW
# ============================================================

def test_allow_path():

    sql = """
    SELECT
        order_status,
        COUNT(*) AS order_count
    FROM fact_orders
    GROUP BY order_status
    """

    result = check_query_resources(
        sql
    )

    print_result(
        "ALLOW PATH",
        result,
    )

    assert result["allowed"] is True

    assert (
        result["decision"]
        == "ALLOW"
    )

    assert (
        result["risk_level"]
        == "LOW"
    )

    assert (
        result["violations"]
        == []
    )


# ============================================================
# WARN
# ============================================================

def test_warn_path():

    sql = """
    SELECT *
    FROM fact_orders
    LIMIT 100
    """

    result = check_query_resources(
        sql
    )

    print_result(
        "WARN PATH",
        result,
    )

    assert result["allowed"] is True

    assert (
        result["decision"]
        == "WARN"
    )

    assert (
        result["risk_level"]
        == "MEDIUM"
    )

    assert (
        "SELECT_STAR"
        in result["violations"]
    )


# ============================================================
# BLOCK
# ============================================================

def test_block_path():

    sql = """
    SELECT
        order_id
    FROM fact_orders
    LIMIT 500000
    """

    result = check_query_resources(
        sql
    )

    print_result(
        "BLOCK PATH",
        result,
    )

    assert result["allowed"] is False

    assert (
        result["decision"]
        == "BLOCK"
    )

    assert (
        result["risk_level"]
        == "HIGH"
    )

    assert (
        "EXCESSIVE_LIMIT"
        in result["violations"]
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    test_allow_path()

    test_warn_path()

    test_block_path()

    print()
    print("=" * 70)
    print(
        "ALL RESOURCE PIPELINE TESTS PASSED"
    )
    print("=" * 70)