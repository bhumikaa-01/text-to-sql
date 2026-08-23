import json
from unittest.mock import patch

import pytest

from agent.semantic_evaluator import (
    _build_prompt,
    _parse_response,
    evaluate_semantics,
)


# ============================================================
# TEST DATA
# ============================================================

QUESTION = "How many orders were cancelled?"

GENERATED_SQL = """
SELECT COUNT(DISTINCT fo.order_id) AS canceled_orders_count
FROM fact_orders fo
WHERE fo.order_status = 'canceled';
"""

RESULTS = [
    {
        "canceled_orders_count": 625
    }
]


# ============================================================
# TEST 1 — Prompt generation
# ============================================================


def test_prompt_contains_required_context():

    prompt = _build_prompt(
        question=QUESTION,
        generated_sql=GENERATED_SQL,
        results=RESULTS,
    )

    assert QUESTION in prompt

    print("Question included: PASS")

    assert GENERATED_SQL in prompt

    print("Generated SQL included: PASS")

    assert "625" in prompt

    print("Actual result included: PASS")

    assert "SEMANTIC SCHEMA" in prompt

    print("Semantic schema included: PASS")

    assert "Do NOT execute SQL." in prompt

    print("Evaluator constraints included: PASS")


# ============================================================
# TEST 2 — Valid JSON response parsing
# ============================================================


def test_parse_valid_response():

    response = json.dumps(
        {
            "is_correct": True,
            "score": 0.95,
            "reason": "The SQL correctly counts cancelled orders.",
            "issues": [],
        }
    )

    result = _parse_response(response)

    assert result["is_correct"] is True

    print("is_correct parsing: PASS")

    assert result["score"] == 0.95

    print("Score parsing: PASS")

    assert (
        result["reason"]
        == "The SQL correctly counts cancelled orders."
    )

    print("Reason parsing: PASS")

    assert result["issues"] == []

    print("Issues parsing: PASS")


# ============================================================
# TEST 3 — Markdown fenced JSON
# ============================================================
def test_parse_markdown_json():

    response = """
```json
{
    "is_correct": true,
    "score": 0.9,
    "reason": "Correct query.",
    "issues": []
}

"""

    result = _parse_response(response)

    assert result["is_correct"] is True

    print("Markdown JSON parsing: PASS")

    assert result["score"] == 0.9

    print("Markdown score parsing: PASS")
# ============================================================
# TEST 4 — Score normalization
# ============================================================

def test_score_normalization():

    response = json.dumps(
    {
        "is_correct": True,
        "score": 5,
        "reason": "High score.",
        "issues": [],
    }
)

    result = _parse_response(response)

    assert result["score"] == 1.0

    print("Upper score normalization: PASS")

    response = json.dumps(
        {
        "is_correct": False,
        "score": -2,
        "reason": "Low score.",
        "issues": [
            "Wrong aggregation"
        ],
    }
)

    result = _parse_response(response)

    assert result["score"] == 0.0

    print("Lower score normalization: PASS")
#============================================================
#TEST 5 — Missing required field
#============================================================

def test_parse_missing_field():

    response = json.dumps(
    {
        "is_correct": True,
        "score": 0.9,
        "reason": "Correct query.",
    }
)

    with pytest.raises(ValueError):

        _parse_response(response)

    print("Missing field validation: PASS")
#============================================================
#TEST 6 — Invalid JSON
#============================================================

def test_parse_invalid_json():

    response = """
{
    this is not valid json
}
"""

    with pytest.raises(ValueError):

        _parse_response(response)

    print("Invalid JSON validation: PASS")
#============================================================
#TEST 7 — Invalid is_correct type
#============================================================

def test_invalid_is_correct_type():

    response = json.dumps(
    {
        "is_correct": "true",
        "score": 0.9,
        "reason": "Correct query.",
        "issues": [],
    }
)

    with pytest.raises(ValueError):

        _parse_response(response)

    print("is_correct type validation: PASS")
#============================================================
#TEST 8 — Invalid issues type
#============================================================

def test_invalid_issues_type():

    response = json.dumps(
    {
        "is_correct": True,
        "score": 0.9,
        "reason": "Correct query.",
        "issues": "none",
    }
)

    with pytest.raises(ValueError):

        _parse_response(response)

    print("Issues type validation: PASS")
#============================================================
#TEST 9 — Empty question
#============================================================

def test_empty_question():

    with pytest.raises(ValueError):

        evaluate_semantics(
        question="",
        generated_sql=GENERATED_SQL,
        results=RESULTS,
    )

print("Empty question validation: PASS")
#============================================================
#TEST 10 — Empty SQL
#============================================================

def test_empty_sql():

    result = evaluate_semantics(
    question=QUESTION,
    generated_sql="",
    results=RESULTS,
)

    assert result["is_correct"] is False

    print("Empty SQL correctness: PASS")

    assert result["score"] == 0.0

    print("Empty SQL score: PASS")

    assert "EMPTY_SQL" in result["issues"]

    print("Empty SQL issue: PASS")
#============================================================
#TEST 11 — Successful evaluator call
#============================================================

def test_successful_evaluation():

    mock_response = type(
    "MockResponse",
    (),
    {
        "text": json.dumps(
            {
                "is_correct": True,
                "score": 1.0,
                "reason": (
                    "The SQL correctly counts "
                    "cancelled orders."
                ),
                "issues": [],
            }
        )
    },
)()

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=mock_response,
    ) as mock_generate:

        result = evaluate_semantics(
            question=QUESTION,
            generated_sql=GENERATED_SQL,
            results=RESULTS,
        )

    assert result["is_correct"] is True

    print("Successful evaluation: PASS")

    assert result["score"] == 1.0

    print("Successful evaluation score: PASS")

    assert result["issues"] == []

    print("Successful evaluation issues: PASS")

    assert mock_generate.called

    print("Gemini evaluator call: PASS")
#============================================================
#TEST 12 — Incorrect semantic evaluation
#============================================================

def test_incorrect_evaluation():

    mock_response = type(
    "MockResponse",
    (),
    {
        "text": json.dumps(
            {
                "is_correct": False,
                "score": 0.2,
                "reason": (
                    "The SQL calculates revenue "
                    "instead of cancelled orders."
                ),
                "issues": [
                    "WRONG_METRIC"
                ],
            }
        )
    },
)()

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=mock_response,
    ):

        result = evaluate_semantics(
            question=QUESTION,
            generated_sql=GENERATED_SQL,
            results=RESULTS,
        )

    assert result["is_correct"] is False

    print("Incorrect evaluation detection: PASS")

    assert result["score"] == 0.2

    print("Incorrect evaluation score: PASS")

    assert "WRONG_METRIC" in result["issues"]

    print("Incorrect evaluation issue: PASS")
#============================================================
#TEST 13 — Empty evaluator response
#============================================================

def test_empty_evaluator_response():

    mock_response = type(
    "MockResponse",
    (),
    {
        "text": ""
    },
)()

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=mock_response,
    ):

        with pytest.raises(ValueError):

            evaluate_semantics(
                question=QUESTION,
                generated_sql=GENERATED_SQL,
                results=RESULTS,
            )

            print("Empty evaluator response handling: PASS")
#============================================================
#TEST 14 — Evaluator returns invalid JSON
#============================================================

def test_evaluator_invalid_json_response():

    mock_response = type(
    "MockResponse",
    (),
    {
        "text": "This is not valid JSON."
    },
)()

    with patch(
    "agent.semantic_evaluator._client.models.generate_content",
    return_value=mock_response,
    ):

        with pytest.raises(ValueError):

            evaluate_semantics(
                question=QUESTION,
                generated_sql=GENERATED_SQL,
                results=RESULTS,
            )

            print("Invalid evaluator JSON handling: PASS")

# ============================================================
# TEST 15 — Realistic revenue evaluation
# ============================================================

def test_revenue_evaluation():

    question = (
        "Which product categories generated "
        "the highest total revenue?"
    )

    sql = """
    SELECT dp.category_name,
           ROUND(
               SUM(fo.order_total_usd),
               2
           ) AS total_revenue
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    WHERE fo.order_status = 'delivered'
    GROUP BY dp.category_name
    ORDER BY total_revenue DESC
    """

    results = [
        {
            "category_name": "health_beauty",
            "total_revenue": 125000.50,
        },
        {
            "category_name": "watches_gifts",
            "total_revenue": 98000.25,
        },
    ]

    mock_response = type(
        "MockResponse",
        (),
        {
            "text": json.dumps(
                {
                    "is_correct": True,
                    "score": 1.0,
                    "reason": (
                        "The SQL correctly calculates "
                        "revenue by product category "
                        "for delivered orders."
                    ),
                    "issues": [],
                }
            )
        },
    )()

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=mock_response,
    ):

        result = evaluate_semantics(
            question=question,
            generated_sql=sql,
            results=results,
        )

    assert result["is_correct"] is True

    print("Revenue semantic evaluation: PASS")

    assert result["score"] == 1.0

    print("Revenue semantic score: PASS")

# ============================================================
# TEST 16 — Wrong metric evaluation
# ============================================================


def test_wrong_metric_evaluation():

    question = (
        "Which product categories generated "
        "the highest total revenue?"
    )

    sql = """
    SELECT dp.category_name,
           COUNT(DISTINCT fo.order_id) AS order_count
    FROM fact_orders fo
    JOIN dim_products dp
        ON fo.product_id = dp.product_id
    GROUP BY dp.category_name
    ORDER BY order_count DESC
    """

    results = [
        {
            "category_name": "health_beauty",
            "order_count": 5000,
        }
    ]

    mock_response = type(
        "MockResponse",
        (),
        {
            "text": json.dumps(
                {
                    "is_correct": False,
                    "score": 0.15,
                    "reason": (
                        "The query counts orders instead "
                        "of calculating total revenue."
                    ),
                    "issues": [
                        "WRONG_METRIC"
                    ],
                }
            )
        },
    )()

    with patch(
        "agent.semantic_evaluator._client.models.generate_content",
        return_value=mock_response,
    ):

        result = evaluate_semantics(
            question=question,
            generated_sql=sql,
            results=results,
        )

    assert result["is_correct"] is False

    print("Wrong metric detection: PASS")

    assert "WRONG_METRIC" in result["issues"]

    print("Wrong metric issue: PASS")

#============================================================
#MAIN
#============================================================

def main():

    print("=" * 70)
    print("SEMANTIC EVALUATOR TEST")
    print("=" * 70)

tests = [
    test_prompt_contains_required_context,
    test_parse_valid_response,
    test_parse_markdown_json,
    test_score_normalization,
    test_parse_missing_field,
    test_parse_invalid_json,
    test_invalid_is_correct_type,
    test_invalid_issues_type,
    test_empty_question,
    test_empty_sql,
    test_successful_evaluation,
    test_incorrect_evaluation,
    test_empty_evaluator_response,
    test_evaluator_invalid_json_response,
    test_revenue_evaluation,
    test_wrong_metric_evaluation,
]

for test in tests:

    print()
    print("-" * 70)
    print(test.__name__)
    print("-" * 70)

    test()

print()
print("=" * 70)
print("ALL SEMANTIC EVALUATOR TESTS PASSED")
print("=" * 70)
