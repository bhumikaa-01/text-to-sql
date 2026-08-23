from agent.confidence import (
    calculate_confidence,
)


def test_high_confidence():

    result = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision="ALLOW",
        execution_success=True,
        result_quality=100,
        table_correct=True,
    )

    print()
    print("=" * 70)
    print("HIGH CONFIDENCE")
    print("=" * 70)
    print(result)

    assert result["score"] == 100.0
    assert result["level"] == "HIGH"


def test_medium_confidence():

    result = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision="WARN",
        execution_success=True,
        result_quality=60,
        table_correct=True,
    )

    print()
    print("=" * 70)
    print("MEDIUM CONFIDENCE")
    print("=" * 70)
    print(result)

    assert (
        70
        <= result["score"]
        < 90
    )

    assert result["level"] == "MEDIUM"


def test_low_confidence_execution_failure():

    result = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision="ALLOW",
        execution_success=False,
        result_quality=0,
        table_correct=False,
    )

    print()
    print("=" * 70)
    print("LOW CONFIDENCE — EXECUTION FAILURE")
    print("=" * 70)
    print(result)

    assert result["score"] <= 40
    assert result["level"] == "LOW"


def test_schema_failure_cap():

    result = calculate_confidence(
        sql_safe=True,
        schema_valid=False,
        resource_decision="ALLOW",
        execution_success=False,
        result_quality=0,
        table_correct=False,
    )

    print()
    print("=" * 70)
    print("SCHEMA FAILURE")
    print("=" * 70)
    print(result)

    assert result["score"] <= 30
    assert result["level"] == "LOW"


def test_resource_block():

    result = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision="BLOCK",
        execution_success=False,
        result_quality=0,
        table_correct=False,
    )

    print()
    print("=" * 70)
    print("RESOURCE BLOCK")
    print("=" * 70)
    print(result)

    assert result["score"] == 0.0
    assert result["level"] == "LOW"

def test_table_incorrectness():

    result = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision="ALLOW",
        execution_success=True,
        result_quality=100,
        table_correct=False,
    )

    print()
    print("=" * 70)
    print("TABLE INCORRECTNESS")
    print("=" * 70)
    print(result)

    assert result["score"] == 90.0
    assert result["level"] == "HIGH"
    assert result["factors"]["table_correctness"] == 0.0

    print("Incorrect table confidence: PASS")


if __name__ == "__main__":

    test_high_confidence()
    test_medium_confidence()
    test_low_confidence_execution_failure()
    test_schema_failure_cap()
    test_resource_block()

    print()
    print("=" * 70)
    print(
        "ALL CONFIDENCE TESTS PASSED"
    )
    print("=" * 70)