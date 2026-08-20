"""
Query result visualization helpers.

Analyzes SQL query results and recommends an appropriate
visualization without modifying the underlying query results.
"""

from typing import Any


# ============================================================
# HELPERS
# ============================================================


def _is_numeric(value: Any) -> bool:
    """Return True when a value is numeric."""

    return isinstance(
        value,
        (int, float),
    ) and not isinstance(
        value,
        bool,
    )


def _get_columns(
    results: list[dict[str, Any]],
) -> list[str]:
    """Return columns from the first result row."""

    if not results:
        return []

    return list(
        results[0].keys()
    )


def _get_numeric_columns(
    results: list[dict[str, Any]],
) -> list[str]:
    """Return columns containing numeric values."""

    columns = _get_columns(
        results
    )

    numeric_columns = []

    for column in columns:

        values = [
            row.get(column)
            for row in results
        ]

        if any(
            _is_numeric(value)
            for value in values
        ):
            numeric_columns.append(
                column
            )

    return numeric_columns


def _looks_like_time_column(
    column: str,
) -> bool:
    """Detect common time/date column names."""

    column_lower = column.lower()

    time_terms = (
        "date",
        "time",
        "month",
        "year",
        "week",
        "day",
    )

    return any(
        term in column_lower
        for term in time_terms
    )


# ============================================================
# MAIN VISUALIZATION RECOMMENDER
# ============================================================


def recommend_visualization(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyze query results and recommend a visualization.

    Returns metadata only. Actual chart rendering is handled
    separately so the SQL pipeline remains independent.
    """

    if not results:

        return {
            "recommended": False,
            "chart_type": None,
            "x_axis": None,
            "y_axis": None,
            "reason": "No results available for visualization.",
        }

    columns = _get_columns(
        results
    )

    numeric_columns = _get_numeric_columns(
        results
    )

    # --------------------------------------------------------
    # Single numeric result → KPI
    # --------------------------------------------------------

    if (
        len(results) == 1
        and len(numeric_columns) == 1
    ):

        return {
            "recommended": True,
            "chart_type": "kpi",
            "x_axis": None,
            "y_axis": numeric_columns[0],
            "reason": (
                "Single numeric result is best "
                "represented as a KPI."
            ),
        }

    # --------------------------------------------------------
    # Time + numeric → Line chart
    # --------------------------------------------------------

    time_columns = [
        column
        for column in columns
        if _looks_like_time_column(
            column
        )
    ]

    if (
        time_columns
        and numeric_columns
    ):

        return {
            "recommended": True,
            "chart_type": "line",
            "x_axis": time_columns[0],
            "y_axis": numeric_columns[0],
            "reason": (
                "Time-based results are suitable "
                "for a line chart."
            ),
        }

    # --------------------------------------------------------
    # Category + numeric → Bar chart
    # --------------------------------------------------------

    non_numeric_columns = [
        column
        for column in columns
        if column not in numeric_columns
    ]

    if (
        non_numeric_columns
        and numeric_columns
    ):

        return {
            "recommended": True,
            "chart_type": "bar",
            "x_axis": non_numeric_columns[0],
            "y_axis": numeric_columns[0],
            "reason": (
                "Categorical and numeric results "
                "are suitable for a bar chart."
            ),
        }

    # --------------------------------------------------------
    # Multiple numeric columns → Bar chart
    # --------------------------------------------------------

    if len(numeric_columns) >= 2:

        return {
            "recommended": True,
            "chart_type": "bar",
            "x_axis": None,
            "y_axis": numeric_columns[0],
            "reason": (
                "Multiple numeric columns can be "
                "compared using a bar chart."
            ),
        }

    # --------------------------------------------------------
    # Unsupported structure
    # --------------------------------------------------------

    return {
        "recommended": False,
        "chart_type": None,
        "x_axis": None,
        "y_axis": None,
        "reason": (
            "Result structure is not suitable "
            "for the supported visualizations."
        ),
    }