from agent.query_guard import check_query_resources


def test_production_resource_policy():

    dangerous_sql = """
    SELECT *
    FROM fact_orders
    LIMIT 500000
    """

    result = check_query_resources(
        dangerous_sql
    )

    print()
    print("=" * 70)
    print("RESOURCE GUARD INTEGRATION TEST")
    print("=" * 70)

    print(
        "Allowed:",
        result["allowed"]
    )

    print(
        "Risk:",
        result["risk_level"]
    )

    print(
        "Violations:",
        result["violations"]
    )

    print(
        "Reason:",
        result["reason"]
    )

    assert result["allowed"] is False

    assert (
        "EXCESSIVE_LIMIT"
        in result["violations"]
    )

    print()
    print(
        "Resource guard integration test passed."
    )


if __name__ == "__main__":
    test_production_resource_policy()