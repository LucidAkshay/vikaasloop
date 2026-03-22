# api/export_router.py
# AGPL v3 - VikaasLoop

import asyncio
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Note: In main.py we included this router with prefix="/api",
# so the route here will become /api/export/hub
router = APIRouter(prefix="/export", tags=["export"])

ALLOWED_MODELS_DIR = Path("models").resolve()


def validate_adapter_path(adapter_path: str) -> Path:
    """
    Security: Ensure the adapter path is inside the models/ directory.
    Uses strict pathlib containment to prevent Windows casing bypasses.
    """
    try:
        resolved = Path(adapter_path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid adapter path.")

    # Native pathlib check prevents Windows drive letter casing bypasses
    try:
        resolved.relative_to(ALLOWED_MODELS_DIR)
    except ValueError:
        logger.warning(f"Path traversal attempt blocked: {adapter_path}")
        raise HTTPException(
            status_code=403,
            detail="Adapter path is outside the allowed models directory.",
        )

    # Path must actually exist
    if not resolved.exists():
        raise HTTPException(
            status_code=404, detail=f"Adapter not found at: {adapter_path}"
        )

    # Must contain expected adapter files
    if not (resolved / "adapter_config.json").exists():
        raise HTTPException(
            status_code=400,
            detail="Not a valid adapter directory — missing adapter_config.json",
        )

    return resolved


class ExportRequest(BaseModel):
    adapter_path: str
    repo_name: str


def _sync_upload(folder_path: str, repo_id: str):
    """Synchronous upload logic to be run in a background thread."""
    from huggingface_hub import HfApi

    api = HfApi()
    # Relies on the locally cached token (huggingface-cli login)
    api.upload_folder(folder_path=folder_path, repo_id=repo_id, repo_type="model")


@router.post("/hub")
async def export_to_hub(request: ExportRequest):
    logger.info(f"Export requested: {request.adapter_path} -> {request.repo_name}")

    # Security: Validate path before doing anything
    safe_path = validate_adapter_path(request.adapter_path)

    # Validate repo name format (must be "username/reponame")
    if not re.match(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+$", request.repo_name):
        raise HTTPException(
            status_code=400,
            detail="Invalid repo name. Must be in format: username/repo-name",
        )

    try:
        # CRITICAL FIX: Offload heavy network I/O to a background thread
        # This prevents the FastAPI event loop from freezing during the upload
        await asyncio.to_thread(_sync_upload, str(safe_path), request.repo_name)

        hub_url = f"https://huggingface.co/{request.repo_name}"
        logger.info(f"Successfully exported adapter to {hub_url}")
        return {"success": True, "hub_url": hub_url}

    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Export to HuggingFace Hub failed: {e}", exc_info=True)

        # Catch authentication errors specifically to give actionable feedback
        if "token" in error_msg or "unauthorized" in error_msg or "401" in error_msg:
            raise HTTPException(
                status_code=401,
                detail="HuggingFace authentication failed. Run 'huggingface-cli login' in your terminal first.",
            )

        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
