# agents/orchestrator.py
# AGPL v3 - VikaasLoop
#
# Orchestrator: coordinates the full DataGen -> Train -> Eval -> Skills loop.
# Owns the ModelManager lifecycle and all agent instances.

import asyncio
import uuid
import json
import logging
import os
import re
from collections import deque
from typing import Dict, Optional

from fastapi import WebSocket, APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from agents.skills_library import SkillsLibrary
from agents.model_manager import ModelManager
from agents.datagen_agent import run_datagen_pipeline as _datagen_pipeline
from agents.training_agent import TrainingAgent
from agents.eval_agent import EvalAgent
from config import settings

logger = logging.getLogger(__name__)

# ===========================================================================
# Path helpers
# ===========================================================================

def sanitize_run_id(run_id: str) -> str:
    if not run_id:
        raise ValueError("run_id cannot be empty.")
    if len(run_id) > 64:
        raise ValueError("run_id too long (max 64 chars).")
    if not re.match(r"^[a-zA-Z0-9_]+$", run_id.replace("-", "_")):
        raise ValueError(
            f"Invalid run_id '{run_id}' — only alphanumeric, hyphens, "
            f"and underscores are allowed."
        )
    return str(run_id)

# ===========================================================================
# DataPartitioner
# ===========================================================================

class DataPartitioner:
    DEFAULT_EVAL_SIZE = 50
    MIN_TRAIN_ROWS = 10

    @classmethod
    def split(
        cls,
        training_data_path: str,
        run_id: str,
        eval_size: int = DEFAULT_EVAL_SIZE,
    ) -> str:
        if not os.path.isfile(training_data_path):
            raise FileNotFoundError(f"Training data not found: {training_data_path}")

        tail: deque = deque(maxlen=eval_size)
        total_count = 0

        with open(training_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tail.append(line)
                    total_count += 1

        if total_count == 0:
            raise ValueError(f"Training data file is empty: {training_data_path}")

        if total_count <= cls.MIN_TRAIN_ROWS + eval_size:
            actual_eval = max(1, total_count // 5)
        else:
            actual_eval = eval_size

        eval_lines = list(tail)[-actual_eval:]
        trim_at = total_count - actual_eval

        eval_dir = os.path.join("data", "eval")
        os.makedirs(eval_dir, exist_ok=True)
        eval_path = os.path.join(eval_dir, f"{run_id}_eval.jsonl")

        with open(eval_path, "w", encoding="utf-8") as f:
            f.writelines(eval_lines)

        tmp_path = training_data_path + ".tmp"
        written = 0
        try:
            with open(training_data_path, "r", encoding="utf-8") as src, \
                 open(tmp_path, "w", encoding="utf-8") as dst:
                for line in src:
                    if line.strip():
                        if written < trim_at:
                            dst.write(line)
                            written += 1
                        else:
                            break
            os.replace(tmp_path, training_data_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        logger.info(
            f"DataPartitioner: {total_count} total rows -> "
            f"{written} training, {len(eval_lines)} eval  [run_id={run_id}]"
        )
        return eval_path

# ===========================================================================
# Thin DataGenAgent shim
# ===========================================================================

class DataGenAgent:
    async def run_datagen_pipeline(
        self,
        task: str,
        hint: str,
        target_count: int = 100,
    ) -> dict:
        return await _datagen_pipeline(task, hint, target_count=target_count)

# ===========================================================================
# Request / response schemas
# ===========================================================================

class LoopStartRequest(BaseModel):
    task_description: str = Field(..., min_length=10, max_length=500)
    base_model: str = Field(default="microsoft/phi-2")
    target_score: float = Field(default=0.7, ge=0.5, le=0.99)
    max_iterations: int = Field(default=5, ge=1, le=20)
    task_type: str = Field(default="general", max_length=50)

class LoopStartResponse(BaseModel):
    loop_id: str
    message: str

# ===========================================================================
# Module level loop state
# ===========================================================================

_loop_queues: Dict[str, asyncio.Queue] = {}
_loop_tasks: Dict[str, asyncio.Task] = {}

# ===========================================================================
# Orchestrator
# ===========================================================================

class Orchestrator:
    def __init__(
        self,
        gemini_api_key: str = None,
        skills_library=None,
        datagen_agent=None,
        training_agent_class=None,
        eval_agent=None,
    ):
        self.gemini_api_key = gemini_api_key or settings.GEMINI_API_KEY
        self.skills_library = skills_library or SkillsLibrary(db_path=settings.SKILLS_DB_PATH)
        self.model_manager = ModelManager()
        self.datagen_agent = datagen_agent or DataGenAgent()
        self.training_agent_class = training_agent_class or TrainingAgent
        self.eval_agent = eval_agent or EvalAgent(
            model_manager=self.model_manager,
            gemini_api_key=self.gemini_api_key,
        )

    def _prepare_workspace(self, loop_id: str) -> None:
        os.makedirs("models", exist_ok=True)
        os.makedirs(os.path.join("data", "generated"), exist_ok=True)
        os.makedirs(os.path.join("data", "eval"), exist_ok=True)
        os.makedirs(os.path.join("data", "training", loop_id), exist_ok=True)

    async def run_loop(
        self,
        loop_id: str,
        task_description: str,
        base_model: str,
        target_score: float,
        max_iterations: int,
        task_type: str = "general",
        queue: asyncio.Queue = None,
    ) -> float:
        loop_id = sanitize_run_id(loop_id)
        display_task = self._get_display_task(task_description)
        logger.info(f"[{loop_id}] Loop starting — task: {display_task!r}")
        self._prepare_workspace(loop_id)

        current_win_rate = 0.0
        iteration = 0

        try:
            while iteration < max_iterations and current_win_rate < target_score:
                iteration += 1

                await self._emit(queue, loop_id, {
                    "type": "iteration_start",
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "win_rate": current_win_rate,
                })

                strategies = await asyncio.to_thread(
                    self.skills_library.get_top_strategies, task_description
                )
                strategy_hint = strategies[0] if strategies else "General improvement"
                await self._emit(queue, loop_id, {
                    "type": "agent_status",
                    "agent": "skills",
                    "message": f"Strategy: {strategy_hint}",
                })

                await self._emit(queue, loop_id, {
                    "type": "agent_status", "agent": "datagen",
                    "message": "Generating training data…",
                })
                gen_metadata = await self.datagen_agent.run_datagen_pipeline(
                    task_description, strategy_hint
                )
                run_id = sanitize_run_id(f"{loop_id}-it{iteration}")
                gen_metadata["run_id"] = run_id

                training_data_path = gen_metadata.get(
                    "filename",
                    os.path.join("data", "generated", f"{run_id}.jsonl"),
                )
                
                await self._validate_gen_count(loop_id, gen_metadata, queue)
                gen_count = gen_metadata.get("count", 0)

                await self._emit(queue, loop_id, {
                    "type": "agent_status", "agent": "datagen",
                    "message": f"Generated {gen_count} pairs.",
                })

                eval_dataset_path = DataPartitioner.split(training_data_path, run_id=run_id)
                
                await self._emit(queue, loop_id, {
                    "type": "agent_status", "agent": "training",
                    "message": f"Fine-tuning {base_model}…",
                })
                trainer = self.training_agent_class(
                    base_model_name=base_model,
                    training_data_path=training_data_path,
                    run_id=run_id,
                )
                train_metadata = await trainer.train()
                await self._emit(queue, loop_id, {
                    "type": "agent_status", "agent": "training",
                    "message": (
                        f"Training complete. "
                        f"Loss: {train_metadata['final_loss']:.4f}  "
                        f"Time: {train_metadata['training_time_seconds']:.0f}s"
                    ),
                })

                await self._emit(queue, loop_id, {
                    "type": "agent_status", "agent": "eval",
                    "message": "Evaluating adapter vs base model…",
                })
                eval_metadata = await self.eval_agent.run_evaluation_async(
                    base_model_name=base_model,
                    adapter_path=train_metadata["adapter_path"],
                    eval_dataset_path=eval_dataset_path,
                    task_description=task_description,
                    loop_id=loop_id,
                    queue=queue,
                )
                current_win_rate = eval_metadata["win_rate"]
                await self._emit(queue, loop_id, {
                    "type": "score_update",
                    "iteration": iteration,
                    "score": current_win_rate,
                    "wins_base": eval_metadata["wins_base"],
                    "wins_finetuned": eval_metadata["wins_finetuned"],
                    "ties": eval_metadata["ties"],
                })

                await self._execute_skills_update(task_description, strategy_hint, task_type, iteration, current_win_rate, queue, loop_id)

            await self._execute_loop_completion(current_win_rate, target_score, train_metadata, queue, loop_id)

        except Exception as exc:
            logger.error(f"[{loop_id}] Loop failed: {exc}", exc_info=True)
            await self._emit(queue, loop_id, {
                "type": "loop_error",
                "message": f"Critical error: {exc}",
            })
            raise

        finally:
            try:
                self.model_manager.release()
                logger.info(f"[{loop_id}] ModelManager released.")
            except Exception as exc:
                logger.warning(f"[{loop_id}] ModelManager release failed (non-critical): {exc}")

        return current_win_rate

    async def _execute_skills_update(self, task_description, strategy_hint, task_type, iteration, current_win_rate, queue, loop_id):
        await asyncio.to_thread(
            self.skills_library.update_strategy_score,
            task_description,
            strategy_hint,
            task_type,
            iteration,
            current_win_rate,
        )
        await self._emit(queue, loop_id, {
            "type": "agent_status", "agent": "skills",
            "message": f"Skills library updated. Win rate: {current_win_rate:.2%}",
        })

    async def _execute_loop_completion(self, current_win_rate, target_score, train_metadata, queue, loop_id):
        reason = (
            "Target score reached!"
            if current_win_rate >= target_score
            else "Max iterations reached."
        )
        await self._emit(queue, loop_id, {
            "type": "loop_complete",
            "final_score": current_win_rate,
            "message": reason,
            "adapter_path": train_metadata.get("adapter_path", ""),
        })
        logger.info(f"[{loop_id}] Loop finished: {reason} final_win_rate={current_win_rate:.4f}")

    def _get_display_task(self, task_description: str) -> str:
        if len(task_description) > 100:
            return task_description[:100] + "..."
        return task_description

    async def _validate_gen_count(self, loop_id: str, gen_metadata: dict, queue) -> None:
        gen_count = gen_metadata.get("count", 0)
        if gen_count < 10:
            msg = f"DataGen failed: only {gen_count} pairs generated (min 10 required)."
            logger.error(f"[{loop_id}] {msg}")
            raise ValueError(msg)

    async def _emit(self, queue: Optional[asyncio.Queue], loop_id: str, data: dict) -> None:
        message = {"loop_id": loop_id, **data}
        logger.debug(f"[{loop_id}] emit type={data.get('type', '?')}")
        if queue is not None:
            await queue.put(message)

# ===========================================================================
# Singleton orchestrator used by the router
# ===========================================================================

orchestrator = Orchestrator()
router = APIRouter()

# ===========================================================================
# POST /api/loop/start
# ===========================================================================

@router.post("/api/loop/start", response_model=LoopStartResponse)
async def start_loop(request: LoopStartRequest):
    raw_id = str(uuid.uuid4()).replace("-", "")[:12]
    loop_id = sanitize_run_id(raw_id)

    queue: asyncio.Queue = asyncio.Queue()
    _loop_queues[loop_id] = queue

    task = asyncio.create_task(
        orchestrator.run_loop(
            loop_id=loop_id,
            task_description=request.task_description,
            base_model=request.base_model,
            target_score=request.target_score,
            max_iterations=request.max_iterations,
            task_type=request.task_type,
            queue=queue,
        )
    )
    _loop_tasks[loop_id] = task

    logger.info(f"Loop {loop_id} queued for task: {request.task_description!r}")
    return LoopStartResponse(loop_id=loop_id, message="Loop started.")

# ===========================================================================
# WebSocket /ws/loop/{loop_id}
# ===========================================================================

@router.websocket("/ws/loop/{loop_id}")
async def websocket_endpoint(websocket: WebSocket, loop_id: str):
    origin = websocket.headers.get("origin", "")
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000",
    ).split(",")
    
    if origin and origin not in allowed_origins:
        await websocket.close(code=1008, reason="Policy Violation")
        return

    token = None
    raw_protocols = websocket.headers.get("sec-websocket-protocol", "")
    if raw_protocols:
        protocols = [p.strip() for p in raw_protocols.split(",")]
        token = protocols[0] if protocols else None

    if not token:
        await websocket.close(code=4001, reason="Missing auth token")
        return

    try:
        import jwt as pyjwt
        payload = pyjwt.decode(token, settings.get_jwt_secret, algorithms=["HS256"])
        if payload.get("scope") != "ws:loop":
            raise ValueError("Invalid scope")
    except Exception:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    queue = _loop_queues.get(loop_id)
    if queue is None:
        await websocket.close(code=4004, reason="Loop not found.")
        return

    await websocket.accept(subprotocol=token)
    logger.info(f"[{loop_id}] WebSocket connected securely.")

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping", "loop_id": loop_id}))
                continue

            await websocket.send_text(json.dumps(msg))

            if msg.get("type") in ("loop_complete", "loop_error"):
                break

    except Exception as exc:
        logger.warning(f"[{loop_id}] WebSocket stream error: {exc}")
    finally:
        _loop_queues.pop(loop_id, None)
        logger.info(f"[{loop_id}] WebSocket closed.")
        try:
            await websocket.close()
        except Exception:
            pass