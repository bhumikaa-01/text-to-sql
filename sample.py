from agent.hitl_guard import check_sql
from agent.schema_validator import validate_sql_schema


def test_sql_safety():

    safe = check_sql(
        "SELECT SUM(order_total_usd) FROM fact_orders"
    )

    assert safe["allowed"] is True

    dangerous = check_sql(
        "DROP TABLE fact_orders"
    )

    assert dangerous["allowed"] is False

    multi_statement = check_sql(
        "SELECT * FROM fact_orders; DROP TABLE fact_orders;"
    )

    assert multi_statement["allowed"] is False


def test_schema_validation():

    valid, error = validate_sql_schema(
        """
        SELECT fo.order_total_usd
        FROM fact_orders fo
        """
    )

    assert valid is True
    assert error is None

    invalid_table, error = validate_sql_schema(
        """
        SELECT *
        FROM employee_records
        """
    )

    assert invalid_table is False

    invalid_column, error = validate_sql_schema(
        """
        SELECT fo.employee_salary
        FROM fact_orders fo
        """
    )

    assert invalid_column is False


if __name__ == "__main__":

    test_sql_safety()
    test_schema_validation()

    print(
        "All SQL guardrail tests passed."
    )