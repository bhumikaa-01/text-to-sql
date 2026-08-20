from agent.result_visualizer import (
    recommend_visualization,
)


def test_kpi():

    results = [
        {
            "total_revenue": 2718328.74
        }
    ]

    result = recommend_visualization(
        results
    )

    assert result["recommended"] is True
    assert result["chart_type"] == "kpi"
    assert result["y_axis"] == "total_revenue"

    print("KPI visualization: PASS")


def test_bar_chart():

    results = [
        {
            "category_name": "Health Beauty",
            "total_revenue": 246626.34,
        },
        {
            "category_name": "Watches Gifts",
            "total_revenue": 233235.40,
        },
    ]

    result = recommend_visualization(
        results
    )

    assert result["recommended"] is True
    assert result["chart_type"] == "bar"
    assert result["x_axis"] == "category_name"
    assert result["y_axis"] == "total_revenue"

    print("Bar chart visualization: PASS")


def test_line_chart():

    results = [
        {
            "month": "2025-01",
            "revenue": 1000,
        },
        {
            "month": "2025-02",
            "revenue": 1500,
        },
    ]

    result = recommend_visualization(
        results
    )

    assert result["recommended"] is True
    assert result["chart_type"] == "line"
    assert result["x_axis"] == "month"
    assert result["y_axis"] == "revenue"

    print("Line chart visualization: PASS")


def test_empty_results():

    result = recommend_visualization(
        []
    )

    assert result["recommended"] is False
    assert result["chart_type"] is None

    print("Empty results handling: PASS")


def test_unsupported_results():

    results = [
        {
            "name": "Alice",
            "category": "Customer",
        },
        {
            "name": "Bob",
            "category": "Customer",
        },
    ]

    result = recommend_visualization(
        results
    )

    assert result["recommended"] is False

    print("Unsupported result handling: PASS")


if __name__ == "__main__":

    print("=" * 70)
    print("QUERY RESULT VISUALIZATION TEST")
    print("=" * 70)

    test_kpi()
    test_bar_chart()
    test_line_chart()
    test_empty_results()
    test_unsupported_results()

    print()
    print("=" * 70)
    print("ALL RESULT VISUALIZATION TESTS PASSED")
    print("=" * 70)