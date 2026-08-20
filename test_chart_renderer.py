from pathlib import Path

from agent.chart_renderer import render_chart


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

    visualization = {
        "recommended": True,
        "chart_type": "bar",
        "x_axis": "category_name",
        "y_axis": "total_revenue",
    }

    result = render_chart(
        results,
        visualization,
    )

    assert result["rendered"] is True
    assert result["chart_type"] == "bar"
    assert Path(
        result["chart_path"]
    ).exists()

    print("Bar chart rendering: PASS")


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

    visualization = {
        "recommended": True,
        "chart_type": "line",
        "x_axis": "month",
        "y_axis": "revenue",
    }

    result = render_chart(
        results,
        visualization,
    )

    assert result["rendered"] is True
    assert result["chart_type"] == "line"
    assert Path(
        result["chart_path"]
    ).exists()

    print("Line chart rendering: PASS")


def test_kpi():

    results = [
        {
            "total_revenue": 2718328.74
        }
    ]

    visualization = {
        "recommended": True,
        "chart_type": "kpi",
        "x_axis": None,
        "y_axis": "total_revenue",
    }

    result = render_chart(
        results,
        visualization,
    )

    assert result["rendered"] is True
    assert result["chart_type"] == "kpi"
    assert Path(
        result["chart_path"]
    ).exists()

    print("KPI rendering: PASS")


def test_no_visualization():

    result = render_chart(
        [],
        {
            "recommended": False,
            "chart_type": None,
        },
    )

    assert result["rendered"] is False
    assert result["chart_path"] is None

    print("No visualization handling: PASS")


if __name__ == "__main__":

    print("=" * 70)
    print("CHART RENDERER TEST")
    print("=" * 70)

    test_bar_chart()
    test_line_chart()
    test_kpi()
    test_no_visualization()

    print()
    print("=" * 70)
    print("ALL CHART RENDERER TESTS PASSED")
    print("=" * 70)