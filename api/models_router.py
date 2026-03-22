# api/models_router.py
# AGPL v3 - VikaasLoop
#
# Endpoint that scans the models/ directory for completed fine-tuned adapters
# and returns metadata for the Model Export view in the dashboard.

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])

MODELS_DIR = "models"


def _scan_adapters() -> List[Dict[str, Any]]:
    """
    Walk models/ looking for directories that contain adapter_config.json.
    Returns a list of adapter metadata dicts sorted newest first.
    """
    results: List[Dict[str, Any]] = []

    if not os.path.isdir(MODELS_DIR):
        return results

    for run_dir in sorted(os.listdir(MODELS_DIR), reverse=True):
        run_path = os.path.join(MODELS_DIR, run_dir)
        if not os.path.isdir(run_path):
            continue

        # Adapter may be directly in the run dir or in a subdirectory called adapter/
        for candidate in [run_path, os.path.join(run_path, "adapter")]:
            config_path = os.path.join(candidate, "adapter_config.json")
            if not os.path.isfile(config_path):
                continue

            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                config = {}

            try:
                stat = os.stat(candidate)
                # Resilient file size calculation
                total_size = 0
                for f in os.listdir(candidate):
                    full_path = os.path.join(candidate, f)
                    if os.path.isfile(full_path):
                        try:
                            total_size += os.path.getsize(full_path)
                        except OSError as e:
                            logger.warning(f"Could not read size of {full_path}: {e}")

                results.append(
                    {
                        "run_id": run_dir,
                        "adapter_path": candidate,
                        "base_model": config.get("base_model_name_or_path", "unknown"),
                        "lora_r": config.get("r", "?"),
                        "created_at": stat.st_mtime,
                        "size_mb": round(total_size / (1024 * 1024), 1),
                    }
                )
                break  # found adapter in this run_dir — no need to check subdirs

            except Exception as exc:
                logger.error(f"Error scanning candidate adapter {candidate}: {exc}")

    return results


@router.get("/completed")
async def list_completed_models():
    """List all fine-tuned adapters available for export."""
    return await asyncio.to_thread(_scan_adapters)
