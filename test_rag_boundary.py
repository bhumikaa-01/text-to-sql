import pytest

from agent.retriever import get_relevant_schema


@pytest.mark.parametrize(
    "question",
    [
        "What is the total revenue?",
        "How much money did the company make?",
        "How many orders were canceled?",
        "Which product generated the most sales?",
        "How many customers do we have?",
        "What was the average order value?",
    ],
)
def test_relevant_questions_retrieve_schema(question):
    """
    Relevant database questions should retrieve schema
    from the RAG index.
    """

    schema = get_relevant_schema(
        question,
        k=3,
    )

    assert schema != ""
    assert "Table:" in schema


@pytest.mark.parametrize(
    "question",
    [
        "What is the capital of France?",
        "What is the average salary of employees?",
        "Tell me a joke?",
    ],
)
def test_irrelevant_questions_do_not_retrieve_schema(question):
    """
    Clearly unrelated questions should be rejected by
    the RAG relevance guard.
    """

    schema = get_relevant_schema(
        question,
        k=3,
    )

    assert schema == ""