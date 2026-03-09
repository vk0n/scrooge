from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("")
async def websocket_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        await websocket.send_json(
            {
                "type": "hello",
                "mode": "mock",
                "message": "Scrooge control WS connected",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        while True:
            message = await websocket.receive_text()
            await websocket.send_json(
                {
                    "type": "echo",
                    "mode": "mock",
                    "received": message,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
    except WebSocketDisconnect:
        return
