from __future__ import annotations

import logging

from fastapi import FastAPI

from routes.config import router as config_router
from routes.control import router as control_router
from routes.health import router as health_router
from routes.logs import router as logs_router
from routes.status import router as status_router
from routes.ws import router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="Scrooge Control API", version="0.1.0")

app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(status_router, prefix="/status", tags=["status"])
app.include_router(logs_router, prefix="/logs", tags=["logs"])
app.include_router(config_router, prefix="/config", tags=["config"])
app.include_router(control_router, prefix="/control", tags=["control"])
app.include_router(ws_router, prefix="/ws", tags=["ws"])


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "scrooge-control-api", "status": "ok"}
