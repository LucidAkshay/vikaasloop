# api/datagen_router.py
# AGPL v3 - VikaasLoop

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.datagen_agent import run_datagen_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datagen", tags=["datagen"])


class GenerateRequest(BaseModel):
    task_description: str = Field(..., min_length=10, max_length=500)
    # num_pairs exposed to the API maps to target_count in the pipeline
    num_pairs: int = Field(default=100, ge=10, le=500)
    strategy_hint: str = Field(default="General improvement", max_length=200)


class GenerateResponse(BaseModel):
    run_id: str
    count: int
    avg_quality_score: float
    strategy_used: str
    filename: str


@router.post("/generate", response_model=GenerateResponse)
async def generate_data(request: GenerateRequest):
    logger.info(
        f"Standalone datagen requested: {request.num_pairs} pairs "
        f"for task: {request.task_description[:50]!r}..."
    )
    try:
        result = await run_datagen_pipeline(
            task_description=request.task_description,
            strategy_hint=request.strategy_hint,
            target_count=request.num_pairs,
        )
        return result
    except Exception as exc:
        logger.error(f"Data generation API failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
