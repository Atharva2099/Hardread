from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
BATTLE_LOGS_DIR = ROOT / "battle_logs"
REPLAY_PATH = BATTLE_LOGS_DIR / "replay_battle.json"

app = FastAPI(title="Hardread")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def load_replay() -> dict:
    if not REPLAY_PATH.exists():
        raise HTTPException(status_code=404, detail="replay_battle.json not found")
    return json.loads(REPLAY_PATH.read_text())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/replay")
def replay() -> JSONResponse:
    return JSONResponse(load_replay())
