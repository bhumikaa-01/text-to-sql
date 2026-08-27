"""
Observability API endpoints for the Text-to-SQL system.
"""

from typing import Any

from fastapi import APIRouter, Query

from agent.telemetry_store import (
    get_recent_events,
    get_summary,
)


router = APIRouter(
    prefix="/api/observability",
    tags=["observability"],
)


@router.get("/summary")
async def observability_summary() -> dict[str, Any]:
    """
    Return aggregate observability metrics.
    """

    return get_summary()


@router.get("/recent")
async def observability_recent(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> list[dict[str, Any]]:
    """
    Return the most recent query telemetry events.
    """

    return get_recent_events(
        limit=limit,
    )