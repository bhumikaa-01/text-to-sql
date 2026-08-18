"""
test_semantic_evaluator.py

Unit tests for the semantic evaluator.

These tests DO NOT call the Gemini API.
Gemini responses are mocked so that testing is:
    - deterministic
    - fast
    - free
    - repeatable
"""

from unittest.mock import patch, MagicMock

from agent.semantic_evaluator import evaluate_semantics


def _mock_gemini_response(response_text: str):
    """
    Build a fake Gemini response.
    """

    response = MagicMock()
    response.text = response_text

    return response


# ============================================================
# CORRECT REVENUE QUERY
# ============================================================

def test_correct_revenue_query():

    fake_response = _mock_gemini_response(
        """
        {
            "is_correct": true,
            "score": 1.0,
            "reason": "The SQL correctly calculates completed revenue.",
            "issues": []
        }
        """
    )

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=fake_response,
    ):

        result = evaluate_semantics(
            question="What is the total revenue?",
            generated_sql="""
                SELECT
                    SUM(order_total_usd) AS total_revenue
                FROM fact_orders
                WHERE order_status = 'delivered'
            """,
            results=[
                {
                    "total_revenue": 2644299.62
                }
            ],
        )

    print()
    print("=" * 70)
    print("CORRECT REVENUE QUERY")
    print("=" * 70)
    print(result)

    assert result["is_correct"] is True
    assert result["score"] == 1.0
    assert result["issues"] == []


# ============================================================
# INCORRECT REVENUE QUERY
# ============================================================

def test_incorrect_revenue_query():

    fake_response = _mock_gemini_response(
        """
        {
            "is_correct": false,
            "score": 0.0,
            "reason": "The SQL calculates average shipping cost instead of total revenue.",
            "issues": [
                "Incorrect column",
                "Incorrect aggregation"
            ]
        }
        """
    )

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=fake_response,
    ):

        result = evaluate_semantics(
            question="What is the total revenue?",
            generated_sql="""
                SELECT
                    AVG(freight_value_usd) AS average_shipping
                FROM fact_orders
            """,
            results=[
                {
                    "average_shipping": 18.42
                }
            ],
        )

    print()
    print("=" * 70)
    print("INCORRECT REVENUE QUERY")
    print("=" * 70)
    print(result)

    assert result["is_correct"] is False
    assert result["score"] == 0.0
    assert len(result["issues"]) > 0


# ============================================================
# MONTHLY REVENUE QUERY
# ============================================================

def test_monthly_revenue_query():

    fake_response = _mock_gemini_response(
        """
        {
            "is_correct": true,
            "score": 1.0,
            "reason": "The SQL correctly calculates completed revenue by month.",
            "issues": []
        }
        """
    )

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=fake_response,
    ):

        result = evaluate_semantics(
            question="Show completed revenue by month.",
            generated_sql="""
                SELECT
                    strftime('%Y-%m', created_at) AS month,
                    SUM(order_total_usd) AS revenue
                FROM fact_orders
                WHERE order_status = 'delivered'
                GROUP BY strftime(
                    '%Y-%m',
                    created_at
                )
            """,
            results=[
                {
                    "month": "2026-01",
                    "revenue": 120000.0,
                },
                {
                    "month": "2026-02",
                    "revenue": 135000.0,
                },
            ],
        )

    print()
    print("=" * 70)
    print("MONTHLY REVENUE QUERY")
    print("=" * 70)
    print(result)

    assert result["is_correct"] is True
    assert result["score"] == 1.0


# ============================================================
# RESPONSE STRUCTURE
# ============================================================

def test_evaluator_response_structure():

    fake_response = _mock_gemini_response(
        """
        {
            "is_correct": false,
            "score": 0.4,
            "reason": "COUNT(*) counts order-item events.",
            "issues": [
                "COUNT(DISTINCT order_id) should be used."
            ]
        }
        """
    )

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=fake_response,
    ):

        result = evaluate_semantics(
            question="How many orders are there?",
            generated_sql="""
                SELECT COUNT(*)
                FROM fact_orders
            """,
            results=[
                {
                    "count": 100
                }
            ],
        )

    print()
    print("=" * 70)
    print("RESPONSE STRUCTURE")
    print("=" * 70)
    print(result)

    assert "is_correct" in result
    assert "score" in result
    assert "reason" in result
    assert "issues" in result

    assert isinstance(
        result["is_correct"],
        bool,
    )

    assert isinstance(
        result["score"],
        float,
    )

    assert isinstance(
        result["reason"],
        str,
    )

    assert isinstance(
        result["issues"],
        list,
    )


# ============================================================
# INVALID JSON RESPONSE
# ============================================================

def test_invalid_json_response():

    fake_response = _mock_gemini_response(
        "This is not valid JSON."
    )

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=fake_response,
    ):

        try:

            evaluate_semantics(
                question="What is the total revenue?",
                generated_sql="""
                    SELECT SUM(order_total_usd)
                    FROM fact_orders
                """,
                results=[
                    {
                        "total_revenue": 1000
                    }
                ],
            )

        except ValueError as exc:

            assert (
                "invalid JSON"
                in str(exc)
            )

        else:

            raise AssertionError(
                "Expected ValueError for invalid JSON."
            )


# ============================================================
# EMPTY SQL
# ============================================================

def test_empty_sql():

    result = evaluate_semantics(
        question="What is the total revenue?",
        generated_sql="",
        results=[],
    )

    print()
    print("=" * 70)
    print("EMPTY SQL")
    print("=" * 70)
    print(result)

    assert result["is_correct"] is False
    assert result["score"] == 0.0
    assert "EMPTY_SQL" in result["issues"]


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":

    test_correct_revenue_query()

    test_incorrect_revenue_query()

    test_monthly_revenue_query()

    test_evaluator_response_structure()

    test_invalid_json_response()

    test_empty_sql()

    print()
    print("=" * 70)
    print("ALL SEMANTIC EVALUATOR UNIT TESTS PASSED")
    print("=" * 70)