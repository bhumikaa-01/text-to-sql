"""
test_sql_correction_helper.py

Tests for the reusable SQL correction helper.
"""

import asyncio
from unittest.mock import AsyncMock

from agent.sql_chain import _correct_sql


# ============================================================
# TEST DATA
# ============================================================

QUESTION = "How many orders were cancelled?"

PREVIOUS_SQL = """
SELECT COUNT(fo.fake_column)
FROM fact_orders fo
"""

VALIDATION_ERROR = (
    "no such column: fo.fake_column"
)

SCHEMA_CONTEXT = """
fact_orders:
    order_id
    order_status
"""


# ============================================================
# TEST 1 — Successful SQL correction
# ============================================================

def test_successful_sql_correction():

    mock_response = type(
        "MockResponse",
        (),
        {
            "content": """
```sql
SELECT COUNT(DISTINCT fo.order_id) AS cancelled_orders
FROM fact_orders fo
WHERE fo.order_status = 'canceled'
"""
        },
    )()

    mock_llm = type(
        "MockLLM",
        (),
        {
            "ainvoke": AsyncMock(
                return_value=mock_response
            )
        },
    )()

    corrected_sql = asyncio.run(
        _correct_sql(
            question=QUESTION,
            previous_sql=PREVIOUS_SQL,
            validation_error=VALIDATION_ERROR,
            schema_context=SCHEMA_CONTEXT,
            llm=mock_llm,
        )
    )

    assert "SELECT" in corrected_sql

    assert "fact_orders" in corrected_sql

    assert "fake_column" not in corrected_sql

    assert "order_id" in corrected_sql

    mock_llm.ainvoke.assert_awaited_once()

    print("Successful SQL correction: PASS")


# ============================================================
# TEST 2 — Empty correction response
# ============================================================

def test_empty_sql_correction():

    mock_response = type(
        "MockResponse",
        (),
        {
            "content": ""
        },
    )()

    mock_llm = type(
        "MockLLM",
        (),
        {
            "ainvoke": AsyncMock(
                return_value=mock_response
            )
        },
    )()

    try:

        asyncio.run(
            _correct_sql(
                question=QUESTION,
                previous_sql=PREVIOUS_SQL,
                validation_error=VALIDATION_ERROR,
                schema_context=SCHEMA_CONTEXT,
                llm=mock_llm,
            )
        )

        assert False, (
            "Expected ValueError for empty correction"
        )

    except ValueError as exc:

        assert (
            str(exc)
            == "SQL correction model returned empty SQL."
        )

    print("Empty SQL correction handling: PASS")


# ============================================================
# TEST 3 — Correction prompt reaches LLM
# ============================================================

def test_correction_prompt_content():

    mock_response = type(
        "MockResponse",
        (),
        {
            "content": (
                "SELECT COUNT(*) "
                "FROM fact_orders"
            )
        },
    )()

    mock_llm = type(
        "MockLLM",
        (),
        {
            "ainvoke": AsyncMock(
                return_value=mock_response
            )
        },
    )()

    asyncio.run(
        _correct_sql(
            question=QUESTION,
            previous_sql=PREVIOUS_SQL,
            validation_error=VALIDATION_ERROR,
            schema_context=SCHEMA_CONTEXT,
            llm=mock_llm,
        )
    )

    mock_llm.ainvoke.assert_awaited_once()

    prompt = (
        mock_llm.ainvoke
        .call_args.args[0]
    )

    assert QUESTION in prompt

    assert PREVIOUS_SQL in prompt

    assert VALIDATION_ERROR in prompt

    assert SCHEMA_CONTEXT in prompt

    print("Correction prompt content: PASS")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SQL CORRECTION HELPER TESTS")
    print("=" * 70)

    test_successful_sql_correction()

    test_empty_sql_correction()

    test_correction_prompt_content()

    print()
    print("=" * 70)
    print("ALL SQL CORRECTION HELPER TESTS PASSED")
    print("=" * 70)