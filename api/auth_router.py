# api/auth_router.py
import time

import jwt
from fastapi import APIRouter

from config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/token")
async def get_token():
    """Generates a short lived JWT for WebSocket authentication."""
    payload = {"scope": "ws:loop", "exp": time.time() + 3600}  # Expires in 1 hour
    token = jwt.encode(payload, settings.get_jwt_secret, algorithm="HS256")
    return {"token": token}
