"""
Chart rendering for Text-to-SQL results.

Converts visualization metadata and query results into
chart files without changing the SQL pipeline.
"""

from pathlib import Path
from typing import Any
import uuid

import matplotlib.pyplot as plt


CHART_DIR = Path("./data/charts")

def _get_chart_path(chart_type: str) -> Path:
    """Create a unique output path for a generated chart."""

    chart_id = uuid.uuid4().hex[:8]

    return (
        CHART_DIR
        / f"query_{chart_type}_{chart_id}.png"
    )

def _ensure_chart_directory() -> None:
    """Create the chart output directory if required."""

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def render_chart(
    results: list[dict[str, Any]],
    visualization: dict[str, Any],
) -> dict[str, Any]:
    """
    Render a chart from SQL results.

    Returns metadata describing the generated chart.
    """

    if not results:

        return {
            "rendered": False,
            "chart_path": None,
            "reason": "No results available.",
        }

    if not visualization.get(
        "recommended",
        False,
    ):

        return {
            "rendered": False,
            "chart_path": None,
            "reason": (
                "Visualization was not recommended."
            ),
        }

    chart_type = visualization.get(
        "chart_type"
    )

    _ensure_chart_directory()

    if chart_type == "kpi":

        return _render_kpi(
            results,
            visualization,
        )

    if chart_type == "bar":

        return _render_bar(
            results,
            visualization,
        )

    if chart_type == "line":

        return _render_line(
            results,
            visualization,
        )

    return {
        "rendered": False,
        "chart_path": None,
        "reason": (
            f"Unsupported chart type: {chart_type}"
        ),
    }


def _render_kpi(
    results: list[dict[str, Any]],
    visualization: dict[str, Any],
) -> dict[str, Any]:
    """Render a single numeric result as a KPI."""

    y_axis = visualization.get(
        "y_axis"
    )

    value = results[0].get(
        y_axis
    )

    figure, axis = plt.subplots(
        figsize=(8, 4)
    )

    axis.text(
        0.5,
        0.5,
        str(value),
        ha="center",
        va="center",
        fontsize=28,
    )

    axis.set_title(
        y_axis
    )

    axis.axis("off")

    chart_path = _get_chart_path(
        "kpi"
    )

    figure.savefig(
        chart_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return {
        "rendered": True,
        "chart_path": str(
            chart_path
        ),
        "chart_type": "kpi",
    }


def _render_bar(
    results: list[dict[str, Any]],
    visualization: dict[str, Any],
) -> dict[str, Any]:
    """Render categorical results as a bar chart."""

    x_axis = visualization.get(
        "x_axis"
    )

    y_axis = visualization.get(
        "y_axis"
    )

    labels = [
        str(row.get(x_axis))
        for row in results
    ]

    values = [
        row.get(y_axis)
        for row in results
    ]

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.bar(
        labels,
        values,
    )

    axis.set_xlabel(
        x_axis
    )

    axis.set_ylabel(
        y_axis
    )

    axis.set_title(
        f"{y_axis} by {x_axis}"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    figure.tight_layout()

    chart_path = _get_chart_path(
        "bar"
    )

    figure.savefig(
        chart_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return {
        "rendered": True,
        "chart_path": str(
            chart_path
        ),
        "chart_type": "bar",
    }


def _render_line(
    results: list[dict[str, Any]],
    visualization: dict[str, Any],
) -> dict[str, Any]:
    """Render time-based results as a line chart."""

    x_axis = visualization.get(
        "x_axis"
    )

    y_axis = visualization.get(
        "y_axis"
    )

    labels = [
        str(row.get(x_axis))
        for row in results
    ]

    values = [
        row.get(y_axis)
        for row in results
    ]

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.plot(
        labels,
        values,
        marker="o",
    )

    axis.set_xlabel(
        x_axis
    )

    axis.set_ylabel(
        y_axis
    )

    axis.set_title(
        f"{y_axis} over {x_axis}"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    figure.tight_layout()

    chart_path = _get_chart_path(
        "line"
    )

    figure.savefig(
        chart_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return {
        "rendered": True,
        "chart_path": str(
            chart_path
        ),
        "chart_type": "line",
    }