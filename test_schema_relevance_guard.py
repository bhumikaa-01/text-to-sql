from agent.schema_relevance_guard import check_schema_relevance


def test_database_question():

    result = check_schema_relevance(
        "What was the average order value for delivered orders?"
    )

    assert result["relevant"] is True

    print("Database question: PASS")
    print(result)


def test_revenue_question():

    result = check_schema_relevance(
        "Which product category generated the highest revenue?"
    )

    assert result["relevant"] is True

    print("Revenue question: PASS")
    print(result)


def test_out_of_scope_question():

    result = check_schema_relevance(
        "What is the capital of France?"
    )

    assert result["relevant"] is False

    print("Out-of-scope question: PASS")
    print(result)


def test_weather_question():

    result = check_schema_relevance(
        "What will the weather be tomorrow?"
    )

    assert result["relevant"] is False

    print("Weather question: PASS")
    print(result)


if __name__ == "__main__":

    test_database_question()
    test_revenue_question()
    test_out_of_scope_question()
    test_weather_question()

    print()
    print("=" * 60)
    print("ALL SCHEMA RELEVANCE TESTS PASSED")
    print("=" * 60)