"""
test_confidence_integration.py

Integration tests for the Confidence / Reliability Score.

Verifies that the confidence engine correctly reflects:

    1. Resource-governance outcome
    2. Deterministic table correctness
    3. Execution success/failure
    4. Result quality
    5. Unverified table correctness

Scenarios:

    ALLOW + correct table
        -> successful execution
        -> table correctness verified
        -> 100 confidence

    WARN + unverified table
        -> successful execution with resource warning
        -> MEDIUM confidence

    BLOCK
        -> query prevented before execution
        -> LOW confidence

    INVALID TABLE
        -> deterministic table verification fails
        -> table correctness points = 0

    UNVERIFIED TABLE
        -> ground truth unavailable
        -> table correctness points = 0

    EMPTY SQL
        -> LOW confidence
"""

from agent.confidence import calculate_confidence
from agent.query_guard import check_query_resources
from agent.table_correctness import check_table_correctness


# ============================================================
# ALLOW PATH + CORRECT TABLE
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

    # --------------------------------------------------------
    # Deterministic table correctness
    # --------------------------------------------------------

    table_check = check_table_correctness(
        sql=sql
    )

    assert table_check["table_correct"] is True
    assert table_check["tables_used"] == [
        "fact_orders"
    ]

    # --------------------------------------------------------
    # Confidence calculation
    # --------------------------------------------------------

    confidence = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision=resource["decision"],
        execution_success=True,
        result_quality=100,
        table_correct=table_check["table_correct"],
    )

    print()
    print("=" * 70)
    print("ALLOW + CORRECT TABLE CONFIDENCE TEST")
    print("=" * 70)

    print("Resource decision:", resource["decision"])
    print("Risk level:", resource["risk_level"])
    print("Table check:", table_check)
    print("Confidence:", confidence)

    assert confidence["level"] == "HIGH"
    assert confidence["score"] == 100
    assert confidence["factors"]["table_correctness"] == 10

    print("ALLOW + correct table confidence: PASS")


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
    assert confidence["factors"]["table_correctness"] == 0

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
# INVALID TABLE PATH
# ============================================================

def test_invalid_table_confidence():

    sql = """
    SELECT
        fo.order_id
    FROM fake_orders fo
    """

    # --------------------------------------------------------
    # Deterministic table correctness
    # --------------------------------------------------------

    table_check = check_table_correctness(
        sql=sql
    )

    assert table_check["table_correct"] is False
    assert "fake_orders" in table_check["invalid_tables"]

    # --------------------------------------------------------
    # Confidence calculation
    # --------------------------------------------------------

    confidence = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision="ALLOW",
        execution_success=True,
        result_quality=100,
        table_correct=table_check["table_correct"],
    )

    print()
    print("=" * 70)
    print("INVALID TABLE CONFIDENCE TEST")
    print("=" * 70)

    print("Table check:", table_check)
    print("Confidence:", confidence)

    # Everything except table correctness passed.
    # Therefore 100 - 10 table points = 90.
    assert confidence["score"] == 90
    assert confidence["level"] == "HIGH"
    assert confidence["factors"]["table_correctness"] == 0

    print("INVALID TABLE confidence: PASS")


# ============================================================
# UNVERIFIED TABLE PATH
# ============================================================

def test_unverified_table_confidence():

    confidence = calculate_confidence(
        sql_safe=True,
        schema_valid=True,
        resource_decision="ALLOW",
        execution_success=True,
        result_quality=100,
        table_correct=None,
    )

    print()
    print("=" * 70)
    print("UNVERIFIED TABLE CONFIDENCE TEST")
    print("=" * 70)

    print("Confidence:", confidence)

    # None means that table correctness is not verified.
    # The system deliberately awards zero points.
    assert confidence["score"] == 90
    assert confidence["level"] == "HIGH"
    assert confidence["factors"]["table_correctness"] == 0

    print("UNVERIFIED TABLE confidence: PASS")


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
    test_invalid_table_confidence()
    test_unverified_table_confidence()
    test_empty_sql_confidence()

    print()
    print("=" * 70)
    print("ALL CONFIDENCE INTEGRATION TESTS PASSED")
    print("=" * 70)