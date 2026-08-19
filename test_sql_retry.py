from agent.sql_retry import (
    MAX_SQL_RETRIES,
    build_correction_prompt,
    build_retry_metadata,
    is_retryable_error,
)


print("=" * 70)
print("SQL RETRY UNIT TEST")
print("=" * 70)


# ------------------------------------------------------------
# Retry configuration
# ------------------------------------------------------------

assert MAX_SQL_RETRIES == 2

print("MAX_SQL_RETRIES: PASS")


# ------------------------------------------------------------
# Retryable errors
# ------------------------------------------------------------

assert is_retryable_error(
    "Unknown columns: revenue"
)

assert is_retryable_error(
    "Unknown table: orders"
)

assert is_retryable_error(
    "Schema validation failed"
)

assert is_retryable_error(
    "SQL syntax error"
)

print("Retryable errors: PASS")


# ------------------------------------------------------------
# Non-retryable errors
# ------------------------------------------------------------

assert not is_retryable_error(
    "SQL safety check failed"
)

assert not is_retryable_error(
    "Resource guard blocked query"
)

assert not is_retryable_error(
    "Gemini 503 Service Unavailable"
)

assert not is_retryable_error(
    "429 RESOURCE_EXHAUSTED"
)

print("Non-retryable errors: PASS")


# ------------------------------------------------------------
# Correction prompt
# ------------------------------------------------------------

prompt = build_correction_prompt(
    question="What is the total revenue?",
    previous_sql="SELECT SUM(revenue) FROM fact_orders;",
    validation_error="Unknown column: revenue",
    schema_context="fact_orders(order_total_usd, order_status)",
)

assert "What is the total revenue?" in prompt
assert "SELECT SUM(revenue)" in prompt
assert "Unknown column: revenue" in prompt
assert "order_total_usd" in prompt

print("Correction prompt: PASS")


# ------------------------------------------------------------
# Retry metadata
# ------------------------------------------------------------

metadata = build_retry_metadata(
    attempt=1,
    error="Unknown column: revenue",
)

assert metadata["attempt"] == 1
assert metadata["max_retries"] == 2
assert metadata["retry_available"] is True

print("Retry metadata: PASS")


print()
print("=" * 70)
print("ALL SQL RETRY UNIT TESTS PASSED")
print("=" * 70)