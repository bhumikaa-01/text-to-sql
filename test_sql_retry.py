"""
test_sql_retry.py

Unit tests for SQL retry classification and correction helpers.
"""

from agent.sql_retry import (
    MAX_SQL_RETRIES,
    build_correction_prompt,
    build_retry_metadata,
    is_retryable_error,
)


# ============================================================
# TEST 1 — Retryable schema errors
# ============================================================

def test_retryable_schema_errors():

    errors = [
        "unknown column: fo.customer_name",
        "unknown table: fake_orders",
        "no such column: fo.customer_name",
        "no such table: fake_orders",
        "schema validation failed",
        "invalid sql generated",
        "syntax error near SELECT",
        "SQL parsing failed",
        "parse error in generated SQL",
    ]

    for error in errors:

        assert is_retryable_error(error) is True

        print(
            f"Retryable error PASS: {error}"
        )


# ============================================================
# TEST 2 — Non-retryable safety/resource errors
# ============================================================

def test_non_retryable_errors():

    errors = [
        "SQL safety violation",
        "unsafe SQL detected",
        "resource guard blocked query",
        "excessive limit",
        "excessive joins",
        "excessive unions",
        "permission denied",
        "rate limit exceeded",
        "HTTP 429",
        "503 service unavailable",
    ]

    for error in errors:

        assert is_retryable_error(error) is False

        print(
            f"Non-retryable error PASS: {error}"
        )


# ============================================================
# TEST 3 — Empty / None errors
# ============================================================

def test_empty_errors():

    assert is_retryable_error(None) is False

    assert is_retryable_error("") is False

    assert is_retryable_error("   ") is False

    print("Empty error handling: PASS")


# ============================================================
# TEST 4 — Case insensitive matching
# ============================================================

def test_case_insensitive_matching():

    assert (
        is_retryable_error(
            "NO SUCH COLUMN: fo.customer_name"
        )
        is True
    )

    assert (
        is_retryable_error(
            "SYNTAX ERROR near SELECT"
        )
        is True
    )

    assert (
        is_retryable_error(
            "RESOURCE GUARD BLOCKED"
        )
        is False
    )

    print("Case-insensitive matching: PASS")


# ============================================================
# TEST 5 — Retry metadata
# ============================================================

def test_retry_metadata():

    metadata = build_retry_metadata(
        attempt=1,
        max_retries=MAX_SQL_RETRIES,
        error="no such column: fo.customer_name",
    )

    assert metadata["attempt"] == 1

    assert metadata["max_retries"] == MAX_SQL_RETRIES

    assert metadata["retry_available"] is True

    assert (
        metadata["error"]
        == "no such column: fo.customer_name"
    )

    print("Retry metadata: PASS")


# ============================================================
# TEST 6 — Retry exhausted
# ============================================================

def test_retry_metadata_exhausted():

    metadata = build_retry_metadata(
        attempt=MAX_SQL_RETRIES + 1,
        max_retries=MAX_SQL_RETRIES,
        error="no such column: fo.customer_name",
    )

    assert (
        metadata["retry_available"]
        is False
    )

    print("Retry exhaustion metadata: PASS")


# ============================================================
# TEST 7 — Correction prompt
# ============================================================

def test_correction_prompt():

    prompt = build_correction_prompt(
        question="How many orders were cancelled?",
        previous_sql=(
            "SELECT COUNT(fo.fake_column) "
            "FROM fact_orders fo"
        ),
        validation_error=(
            "no such column: fo.fake_column"
        ),
        schema_context=(
            "fact_orders(order_id, order_status)"
        ),
    )

    assert (
        "How many orders were cancelled?"
        in prompt
    )

    assert (
        "fo.fake_column"
        in prompt
    )

    assert (
        "no such column: fo.fake_column"
        in prompt
    )

    assert (
        "fact_orders(order_id, order_status)"
        in prompt
    )

    assert (
        "Return SQL only."
        in prompt
    )

    print("Correction prompt generation: PASS")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SQL RETRY UNIT TESTS")
    print("=" * 70)

    test_retryable_schema_errors()
    test_non_retryable_errors()
    test_empty_errors()
    test_case_insensitive_matching()
    test_retry_metadata()
    test_retry_metadata_exhausted()
    test_correction_prompt()

    print()
    print("=" * 70)
    print("ALL SQL RETRY TESTS PASSED")
    print("=" * 70)