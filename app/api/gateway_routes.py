"""
PromptShield X - gateway API (user chat mode + admin review mode) and the two web apps.

User:   POST /api/chat              message + optional files / URLs -> pipeline -> (LLM)
        GET  /api/chat/{id}         poll a request (e.g. waiting for admin review)
        GET  /api/chat/history      the user's own requests
Admin:  header X-Admin-Token (env PS_ADMIN_TOKEN; local default "admin", flagged in the UI)
        GET  /api/admin/requests    filter by status / decision / attack type / user / text
        GET  /api/admin/requests/{id}
        POST /api/admin/requests/{id}/approve | /reject   {note}
        GET  /api/admin/stats
Pages:  /app (user), /admin (admin console)
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core import gateway

router = APIRouter()
UI_DIR = Path("dashboard/templates")
MAX_FILE_BYTES = 15 * 1024 * 1024


def _admin_token() -> str:
    return os.environ.get("PS_ADMIN_TOKEN", "admin")


def _require_admin(token: Optional[str]):
    if not token or token != _admin_token():
        raise HTTPException(status_code=401, detail="Invalid admin token")


def _public(rec: dict) -> dict:
    """What the end user may see: no internal documents dump."""
    out = {k: rec.get(k) for k in ("id", "created_at", "prompt", "attachments", "decision", "status", "risk_score",
                                   "attack_type", "technique", "explanation", "llm_response", "llm_meta", "review_note")}
    p = rec.get("pipeline") or {}
    out["pipeline"] = {k: p.get(k) for k in gateway.STAGES + ["timings_ms"]}
    return out


@router.post("/api/chat")
async def chat(message: str = Form(""), user_id: str = Form("user"), urls: str = Form("[]"),
               files: List[UploadFile] = File(default=[])):
    if not message.strip() and not files and urls in ("", "[]"):
        raise HTTPException(status_code=400, detail="Send a message, a file or a URL")
    blobs = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"{f.filename} is larger than 15 MB")
        blobs.append((f.filename or "upload", data))
    try:
        url_list = [u for u in json.loads(urls or "[]") if isinstance(u, str)]
    except json.JSONDecodeError:
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
    rec = gateway.process(message, user_id=user_id[:64] or "user", files=blobs, urls=url_list)
    return _public(rec)


@router.get("/api/chat/history")
def history(user_id: str = "user", limit: int = 50):
    return {"items": [_public(r) for r in gateway.query(user_id=user_id, limit=limit)]}


@router.get("/api/chat/{req_id}")
def chat_status(req_id: str):
    rec = gateway.get(req_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return _public(rec)


class ReviewBody(BaseModel):
    note: str = ""


@router.get("/api/admin/requests")
def admin_list(status: Optional[str] = None, decision: Optional[str] = None, attack_type: Optional[str] = None,
               user_id: Optional[str] = None, q: Optional[str] = None, limit: int = 200,
               x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return {"items": gateway.query(status, decision, attack_type, user_id, q, limit)}


@router.get("/api/admin/requests/{req_id}")
def admin_get(req_id: str, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    rec = gateway.get(req_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@router.post("/api/admin/requests/{req_id}/approve")
def admin_approve(req_id: str, body: ReviewBody, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    rec = gateway.review(req_id, True, "admin", body.note)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@router.post("/api/admin/requests/{req_id}/reject")
def admin_reject(req_id: str, body: ReviewBody, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    rec = gateway.review(req_id, False, "admin", body.note)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@router.get("/api/admin/stats")
def admin_stats(x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return {**gateway.stats(), "default_token": _admin_token() == "admin"}


def _page(name: str) -> HTMLResponse:
    path = UI_DIR / name
    if not path.exists():
        return HTMLResponse(f"<h1>{name} not found</h1>", status_code=404)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.get("/app", response_class=HTMLResponse)
def user_app():
    return _page("app.html")


@router.get("/admin", response_class=HTMLResponse)
def admin_app():
    return _page("admin.html")


@router.get("/api/gateway/info")
def gateway_info():
    """Public, non-secret configuration shown in the user app header."""
    from app.core.llm_client import provider_info
    info = provider_info()
    return {"llm_provider": info["provider"], "llm_model": info["model"], "policy": gateway.policy(),
            "stages": gateway.STAGES,
            "modalities": ["TEXT", "PDF", "CSV-XLSX", "IMAGE", "WEB PAGE", "GITHUB"]}
