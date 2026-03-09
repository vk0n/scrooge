from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.config import router as config_router
from routes.control import router as control_router
from routes.health import router as health_router
from routes.logs import router as logs_router
from routes.status import router as status_router
from routes.ws import router as ws_router
from services.auth_service import require_http_auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="Scrooge Control API", version="0.1.0")

_cors_origins_env = os.getenv("SCROOGE_GUI_CORS_ORIGINS", "")
if _cors_origins_env.strip():
    allowed_origins = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(status_router, prefix="/api/status", tags=["status"], dependencies=[Depends(require_http_auth)])
app.include_router(logs_router, prefix="/api/logs", tags=["logs"], dependencies=[Depends(require_http_auth)])
app.include_router(config_router, prefix="/api/config", tags=["config"], dependencies=[Depends(require_http_auth)])
app.include_router(control_router, prefix="/api/control", tags=["control"])
app.include_router(ws_router, prefix="/ws", tags=["ws"])


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "scrooge-control-api", "status": "ok"}
