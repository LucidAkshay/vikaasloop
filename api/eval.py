# api/eval.py
# AGPL v3 - VikaasLoop
#
# REST endpoints for the Eval Agent Dashboard:
#   POST /api/eval/run         - run a standalone evaluation
#   GET  /api/eval/results     - list all past experiments
#   GET  /api/eval/results/:id - detail for one experiment

import uuid
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

import database
from agents.model_manager import ModelManager
from agents.eval_agent import EvalAgent
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/eval", tags=["eval"])

# ---------------------------------------------------------------------------
# Module-level ModelManager + EvalAgent for standalone /run calls.
# The loop-driven evaluations use the Orchestrator's own instances.
# ---------------------------------------------------------------------------
_model_manager = ModelManager()
_eval_agent = EvalAgent(
    model_manager=_model_manager,
    gemini_api_key=settings.GEMINI_API_KEY,
)


class RunEvalRequest(BaseModel):
    base_model_name: Optional[str] = None
    adapter_path: Optional[str] = None
    test_prompts_path: Optional[str] = None
    task_description: str = ""


async def _run_and_save(
    run_id: str,
    base_model: str,
    adapter_path: str,
    test_prompts_path: str,
    task_description: str,
) -> None:
    """Background task: run evaluation, persist results, and guarantee VRAM cleanup."""
    try:
        result = await _eval_agent.run_evaluation_async(
            base_model_name=base_model,
            adapter_path=adapter_path,
            eval_dataset_path=test_prompts_path,
            task_description=task_description,
        )

        prev = database.get_latest_experiment()
        score_delta = (
            result["win_rate"] - prev["win_rate"]
            if prev and "win_rate" in prev
            else None
        )

        database.save_experiment(
            run_id=run_id,
            base_model=base_model,
            finetuned_model=adapter_path,
            win_rate=result["win_rate"],
            score_delta=score_delta,
            total_comparisons=result["total_evaluated"],
            a_wins=result["wins_base"],
            b_wins=result["wins_finetuned"],
            ties=result["ties"],
            comparisons=result["sample_comparisons"],
            task_description=task_description,
        )
        logger.info(f"Eval run {run_id} saved. win_rate={result['win_rate']:.3f}")

    except Exception as exc:
        logger.error(f"Background eval run {run_id} failed: {exc}", exc_info=True)
    finally:
        # CRITICAL FIX: Guarantee VRAM is released even if evaluation crashes
        try:
            _model_manager.release()
            logger.info(f"Standalone Eval run {run_id} VRAM released.")
        except Exception as cleanup_exc:
            logger.warning(f"Failed to release VRAM after standalone eval: {cleanup_exc}")


@router.post("/run")
async def run_eval(request: RunEvalRequest, background_tasks: BackgroundTasks):
    """
    Start a standalone evaluation in the background.
    Returns immediately with a run_id.
    """
    base_model = request.base_model_name or settings.BASE_MODEL_NAME
    adapter = request.adapter_path or settings.ADAPTER_PATH
    prompts = request.test_prompts_path or settings.TEST_PROMPTS_PATH
    run_id = str(uuid.uuid4()).replace("-", "")[:12]

    background_tasks.add_task(
        _run_and_save,
        run_id=run_id,
        base_model=base_model,
        adapter_path=adapter,
        test_prompts_path=prompts,
        task_description=request.task_description,
    )
    return {"status": "evaluation started", "run_id": run_id}


@router.get("/results")
async def get_results():
    return await asyncio.to_thread(database.get_experiments)


@router.get("/results/{run_id}")
async def get_result_detail(run_id: str):
    details = await asyncio.to_thread(database.get_experiment_details, run_id)
    if not details:
        raise HTTPException(status_code=404, detail="Run not found")
    return details