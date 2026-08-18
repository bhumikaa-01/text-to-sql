"""
test_confidence_integration.py

Integration tests for the Confidence / Reliability Score.

Verifies that the confidence engine correctly reflects
the resource-governance outcome of a query.

Scenarios:

    ALLOW
        -> successful execution
        -> HIGH confidence

    WARN
        -> successful execution with resource warning
        -> MEDIUM confidence

    BLOCK
        -> query prevented before execution
        -> LOW confidence
"""

from agent.confidence import calculate_confidence
from agent.query_guard import check_query_resources


# ============================================================
# ALLOW PATH
# ============================================================

def test_allow_confidence():

    sql = """
    SELECT
        order_id
    FROM fact_orders
    LIMIT 100
    """

    resource = check_query_resources(sql)

    assert resource["allowed"] is True
    assert resource["decision"] == "ALLOW"

    confidence = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision=resource["decision"],
        execution_success=True,
        result_quality=100,
        table_correct=None,
    )

    print()
    print("=" * 70)
    print("ALLOW CONFIDENCE TEST")
    print("=" * 70)

    print("Resource decision:", resource["decision"])
    print("Risk level:", resource["risk_level"])
    print("Confidence:", confidence)

    assert confidence["level"] == "HIGH"
    assert confidence["score"] == 90

    print("ALLOW confidence: PASS")


# ============================================================
# WARN PATH
# ============================================================

def test_warn_confidence():

    sql = """
    SELECT
        order_id
    FROM fact_orders
    LIMIT 800
    """

    resource = check_query_resources(sql)

    assert resource["allowed"] is True
    assert resource["decision"] == "WARN"

    confidence = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision=resource["decision"],
        execution_success=True,
        result_quality=100,
        table_correct=None,
    )

    print()
    print("=" * 70)
    print("WARN CONFIDENCE TEST")
    print("=" * 70)

    print("Resource decision:", resource["decision"])
    print("Risk level:", resource["risk_level"])
    print("Violations:", resource["violations"])
    print("Confidence:", confidence)

    assert confidence["level"] == "MEDIUM"
    assert confidence["score"] == 85

    print("WARN confidence: PASS")


# ============================================================
# BLOCK PATH
# ============================================================

def test_block_confidence():

    sql = """
    SELECT
        order_id
    FROM fact_orders
    LIMIT 500000
    """

    resource = check_query_resources(sql)

    assert resource["allowed"] is False
    assert resource["decision"] == "BLOCK"

    confidence = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision=resource["decision"],
        execution_success=False,
        result_quality=0,
        table_correct=None,
    )

    print()
    print("=" * 70)
    print("BLOCK CONFIDENCE TEST")
    print("=" * 70)

    print("Resource decision:", resource["decision"])
    print("Risk level:", resource["risk_level"])
    print("Violations:", resource["violations"])
    print("Confidence:", confidence)

    assert confidence["level"] == "LOW"
    assert confidence["score"] == 0

    print("BLOCK confidence: PASS")


# ============================================================
# EMPTY SQL PATH
# ============================================================

def test_empty_sql_confidence():

    confidence = calculate_confidence(
        sql_safe=False,
        schema_valid=False,
        resource_decision="BLOCK",
        execution_success=False,
        result_quality=0,
        table_correct=None,
    )

    print()
    print("=" * 70)
    print("EMPTY SQL CONFIDENCE TEST")
    print("=" * 70)

    print("Confidence:", confidence)

    assert confidence["level"] == "LOW"
    assert confidence["score"] == 0

    print("EMPTY SQL confidence: PASS")


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":

    test_allow_confidence()
    test_warn_confidence()
    test_block_confidence()
    test_empty_sql_confidence()

    print()
    print("=" * 70)
    print("ALL CONFIDENCE INTEGRATION TESTS PASSED")
    print("=" * 70)