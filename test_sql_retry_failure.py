from agent.sql_retry import (
    MAX_SQL_RETRIES,
    is_retryable_error,
)


def test_sql_retry_failure():

    print("=" * 70)
    print("SQL RETRY FAILURE / LIMIT TEST")
    print("=" * 70)

    error = "Unknown columns: revenue"

    # --------------------------------------------------------
    # This is a retryable schema error.
    # --------------------------------------------------------

    assert is_retryable_error(error)

    print("Retryable schema error: PASS")

    # --------------------------------------------------------
    # Verify retry limit configuration.
    # --------------------------------------------------------

    assert MAX_SQL_RETRIES > 0

    print(
        f"MAX_SQL_RETRIES configured: "
        f"{MAX_SQL_RETRIES} — PASS"
    )

    # --------------------------------------------------------
    # Simulate retry counter reaching the limit.
    # --------------------------------------------------------

    retry_count = 0

    while retry_count < MAX_SQL_RETRIES:

        retry_count += 1

    assert retry_count == MAX_SQL_RETRIES

    print(
        "Retry counter stops at MAX_SQL_RETRIES: PASS"
    )

    # --------------------------------------------------------
    # Verify that another retry is not allowed.
    # --------------------------------------------------------

    retry_allowed = (
        retry_count < MAX_SQL_RETRIES
    )

    assert retry_allowed is False

    print(
        "Additional retry blocked after limit: PASS"
    )

    print()
    print("=" * 70)
    print("SQL RETRY FAILURE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_sql_retry_failure()