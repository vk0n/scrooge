from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.config_service import CONFIG_PATH, load_config

router = APIRouter()


@router.get("")
def get_config() -> dict[str, object]:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"config": config, "path": str(CONFIG_PATH)}
