# utils/websocket_manager.py
# AGPL v3 - VikaasLoop
#
# Manages WebSocket connections for streaming training loss to the dashboard.

import asyncio
import logging
from typing import Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Tracks active WebSocket connections keyed by run_id.
    Used by TrainingAgent to stream per-step loss values to the frontend.
    """

    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(
        self, run_id: str, websocket: WebSocket, subprotocol: str = None
    ) -> None:
        """Accepts the WebSocket connection, echoing the requested subprotocol if provided."""
        await websocket.accept(subprotocol=subprotocol)
        self._connections.setdefault(run_id, []).append(websocket)
        logger.debug(
            f"[{run_id}] Training WebSocket connected. "
            f"Total: {len(self._connections[run_id])}"
        )

    def disconnect(self, run_id: str, websocket: WebSocket) -> None:
        conns = self._connections.get(run_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            self._connections.pop(run_id, None)
        logger.debug(f"[{run_id}] Training WebSocket disconnected.")

    async def broadcast(self, run_id: str, message: dict) -> None:
        """
        Send a message to all active connections for this run_id.
        Dead/closed connections are detected and removed automatically.
        """
        conns = self._connections.get(run_id, [])
        if not conns:
            return

        dead: List[WebSocket] = []
        for ws in list(conns):
            try:
                await ws.send_json(message)
            except Exception as exc:
                logger.debug(f"[{run_id}] Removing dead WebSocket: {exc}")
                dead.append(ws)

        for ws in dead:
            self.disconnect(run_id, ws)


ws_manager = WebSocketManager()
