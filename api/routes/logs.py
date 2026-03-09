from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from services.log_service import LOG_PATH, read_last_log_lines

router = APIRouter()


@router.get("")
def get_logs(lines: int = Query(default=200, ge=1, le=5000)) -> dict[str, object]:
    try:
        tail_lines = read_last_log_lines(lines)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "path": str(LOG_PATH),
        "requested_lines": lines,
        "returned_lines": len(tail_lines),
        "lines": tail_lines,
    }
