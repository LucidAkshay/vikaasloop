# api/skills_router.py
# AGPL v3 - VikaasLoop
#
# REST endpoints for the Skills Library — used by the dashboard to display
# stored strategies and allow JSON export.

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import closing
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from agents.skills_library import SkillsLibrary
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/skills", tags=["skills"])

# Shared instance — same DB as the Orchestrator's SkillsLibrary
_library = SkillsLibrary(db_path=settings.SKILLS_DB_PATH)


def _get_strategies(task_type: Optional[str] = None, search: Optional[str] = None):
    """Fetch strategies directly from SQLite with native SQL filtering."""
    if not os.path.exists(settings.SKILLS_DB_PATH):
        return []

    with closing(sqlite3.connect(settings.SKILLS_DB_PATH, timeout=10.0)) as conn:
        conn.row_factory = sqlite3.Row

        query = """
            SELECT id, strategy_name, task_type, win_rate, 
                   iteration_number, task_description, created_at 
            FROM strategies 
            WHERE 1=1
        """
        params = []

        if task_type:
            query += " AND LOWER(task_type) = LOWER(?)"
            params.append(task_type)

        if search:
            query += (
                " AND (LOWER(strategy_name) LIKE ? OR LOWER(task_description) LIKE ?)"
            )
            search_term = f"%{search.lower()}%"
            params.extend([search_term, search_term])

        query += " ORDER BY win_rate DESC"

        try:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.error(f"Skills list query failed: {exc}", exc_info=True)
            return []


@router.get("/list")
async def list_strategies(
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    search: Optional[str] = Query(
        None, description="Search in strategy name or description"
    ),
):
    """
    Return stored strategies sorted by win rate.
    Uses native SQL filtering for high performance at scale.
    """
    return await asyncio.to_thread(_get_strategies, task_type, search)


@router.get("/export")
async def export_strategies():
    """Download all strategies as a JSON file."""
    # Pass None to get all strategies without filters
    rows = await asyncio.to_thread(_get_strategies)
    payload = json.dumps(rows, indent=2, ensure_ascii=False)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=skills_library.json"},
    )


@router.get("/top")
async def top_strategies(
    task_description: str = Query(..., min_length=5),
    top_k: int = Query(default=3, ge=1, le=10),
):
    """Return the top-k strategies most relevant to a task description."""
    results = await asyncio.to_thread(
        _library.get_top_strategies, task_description, None, top_k
    )
    return {"strategies": results}
