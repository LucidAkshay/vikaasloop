# api/training_router.py
# AGPL v3 - VikaasLoop

import os
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
import jwt

from utils.websocket_manager import ws_manager
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/training/{run_id}")
async def training_websocket(websocket: WebSocket, run_id: str):
    """
    Clients connect here to receive live training loss values.
    Auth: JWT token passed as the first WebSocket subprotocol.
    """
    origin = websocket.headers.get("origin", "")
    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000",
    ).split(",")
    
    if origin and origin not in allowed_origins:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    raw = websocket.headers.get("sec-websocket-protocol", "")
    token = [p.strip() for p in raw.split(",")][0] if raw else None

    if not token:
        await websocket.close(code=4001, reason="Missing auth token")
        return

    try:
        payload = jwt.decode(token, settings.get_jwt_secret, algorithms=["HS256"])
        if payload.get("scope") != "ws:loop":
            raise ValueError("Invalid scope")
    except Exception:
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # CRITICAL FIX: Pass the token as the subprotocol so strict browsers don't drop the connection
    await ws_manager.connect(run_id, websocket, subprotocol=token)
    logger.info(f"[{run_id}] Training WebSocket authenticated and connected.")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug(f"[{run_id}] Training WebSocket closed: {exc}")
    finally:
        ws_manager.disconnect(run_id, websocket)
        logger.info(f"[{run_id}] Training WebSocket disconnected.")