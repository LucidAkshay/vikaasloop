# main.py
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI startup and shutdown logic."""
    logger.info("VikaasLoop starting up...")

    try:
        database.init_db()
        logger.info("Database initialized successfully.")
    except Exception as exc:
        logger.error(f"Database initialization failed: {exc}")

    try:
        import google.genai
        from peft import LoraConfig
        from transformers import AutoTokenizer

        logger.info("Critical SDKs verified.")
    except ImportError as err:
        logger.error(f"MISSING DEPENDENCY: {err}. Run pip install -r requirements.txt")

    yield

    logger.info("VikaasLoop shutting down. Releasing resources.")


app = FastAPI(title="VikaasLoop Agent Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["system"])
async def health_check():
    import torch

    return {
        "status": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "db_connected": True,
    }


@app.post("/api/models/unload", tags=["system"])
async def unload_models():
    try:
        from agents.orchestrator import orchestrator
        from agents.training_agent import TRAINING_LOCK

        async with TRAINING_LOCK:

            def _release_vram():
                orchestrator.model_manager.release()
                import gc

                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            await asyncio.to_thread(_release_vram)

        return {"status": "success", "message": "Models unloaded and VRAM cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


from agents.orchestrator import router as orchestrator_router
from api.auth_router import router as auth_router
from api.datagen_router import router as datagen_router
from api.eval import router as eval_router
from api.export_router import router as export_router
from api.models_router import router as models_router
from api.skills_router import router as skills_router
from api.training_router import router as training_router

app.include_router(auth_router)
app.include_router(datagen_router)
app.include_router(eval_router)
app.include_router(export_router, prefix="/api")
app.include_router(training_router)
app.include_router(skills_router)
app.include_router(models_router)
app.include_router(orchestrator_router)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
