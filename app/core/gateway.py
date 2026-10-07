"""
PromptShield X - security gateway in front of any LLM.

    TEXT / PDF / CSV-XLSX / IMAGE / WEB PAGE / GITHUB
      -> Content Extraction -> Normalization -> Detection -> Classification
      -> Risk Engine -> Action Engine -> Explanation + Audit -> (core LLM)

Decisions:
  ALLOW    forwarded to the core LLM as-is
  REWRITE  the attack fragment is removed and the cleaned prompt is forwarded
  REVIEW   held until an admin approves (then forwarded) or rejects it; used when only one signal fires
  BLOCK    stopped; for user prompts only when a detection rule AND the model agree
Policy (PS_POLICY): balanced (default) | strict (REWRITE -> REVIEW) | monitor (log only, always forward).
"""

import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.init_db import DB_PATH
from app.core import calibrated_risk
from app.core.llm_client import complete, provider_info
from app.core.risk_engine import compute_risk_score
from app.core.settings import settings
from app.modules.attack_type import classify_attack_type
from app.modules.pattern_scanner import scan_prompt
from app.modules.sanitizer import extract_hidden, sanitize
from app.modules.semantic_classifier import classify_prompt

STAGES = ["extraction", "normalization", "detection", "classification", "risk", "action", "explanation"]
SEVERITY = {"ALLOW": 0, "REWRITE": 1, "REVIEW": 2, "BLOCK": 3}

SCHEMA = """
CREATE TABLE IF NOT EXISTS gateway_requests (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    user_id TEXT,
    prompt TEXT NOT NULL,
    attachments TEXT,
    decision TEXT NOT NULL,
    status TEXT NOT NULL,
    risk_score INTEGER NOT NULL,
    attack_type TEXT,
    technique TEXT,
    explanation TEXT,
    pipeline TEXT,
    forwarded_prompt TEXT,
    documents TEXT,
    llm_response TEXT,
    llm_meta TEXT,
    reviewed_by TEXT,
    review_note TEXT,
    reviewed_at TEXT
);
"""


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat()


def _row(r) -> Dict[str, Any]:
    d = dict(r)
    for k in ("attachments", "pipeline", "documents", "llm_meta"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    return d


def save(rec: Dict[str, Any]):
    conn = _db()
    try:
        cols = ["id", "created_at", "user_id", "prompt", "attachments", "decision", "status", "risk_score",
                "attack_type", "technique", "explanation", "pipeline", "forwarded_prompt", "documents",
                "llm_response", "llm_meta", "reviewed_by", "review_note", "reviewed_at"]
        vals = [json.dumps(rec[c]) if c in ("attachments", "pipeline", "documents", "llm_meta") and rec.get(c) is not None
                else rec.get(c) for c in cols]
        conn.execute(f"INSERT OR REPLACE INTO gateway_requests ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        conn.commit()
    finally:
        conn.close()


def get(req_id: str) -> Optional[Dict[str, Any]]:
    conn = _db()
    try:
        r = conn.execute("SELECT * FROM gateway_requests WHERE id = ?", (req_id,)).fetchone()
        return _row(r) if r else None
    finally:
        conn.close()


def query(status: Optional[str] = None, decision: Optional[str] = None, attack_type: Optional[str] = None,
          user_id: Optional[str] = None, q: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    sql, args = "SELECT * FROM gateway_requests WHERE 1=1", []
    for col, val in (("status", status), ("decision", decision), ("attack_type", attack_type), ("user_id", user_id)):
        if val:
            sql += f" AND {col} = ?"
            args.append(val)
    if q:
        sql += " AND (prompt LIKE ? OR explanation LIKE ?)"
        args += [f"%{q}%", f"%{q}%"]
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(int(limit))
    conn = _db()
    try:
        return [_row(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def stats() -> Dict[str, Any]:
    conn = _db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0]
        by = lambda col: {r[0] or "none": r[1] for r in conn.execute(f"SELECT {col}, COUNT(*) FROM gateway_requests GROUP BY {col}")}
        daily = [{"day": r[0], "total": r[1], "blocked": r[2]} for r in conn.execute(
            "SELECT substr(created_at,1,10) d, COUNT(*), SUM(decision='BLOCK') FROM gateway_requests GROUP BY d ORDER BY d DESC LIMIT 14")]
        return {"total": total, "by_decision": by("decision"), "by_status": by("status"),
                "by_attack_type": by("attack_type"), "daily": list(reversed(daily)), "llm": provider_info(),
                "policy": policy()}
    finally:
        conn.close()


def policy() -> str:
    return os.environ.get("PS_POLICY", "balanced").lower()


# ----------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------
TEXT_EXT = {".txt", ".md", ".json", ".log"}
CODE_EXT = {".py", ".js", ".ts", ".java", ".go", ".rb", ".cs", ".php", ".cpp", ".c", ".sh"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}


def _extract(name: str, data: bytes, kind: str) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Returns (modality, segments[{text, location, hidden, signals}], pdf_report_or_None)."""
    ext = Path(name.lower()).suffix
    if kind == "url":
        from app.modules.extractors.github_extractor import GitHubExtractor
        from app.modules.extractors.url_extractor import URLExtractor
        url = data.decode("utf-8")
        modality = "GITHUB" if "github.com" in url else "WEB"
        res = (GitHubExtractor() if modality == "GITHUB" else URLExtractor()).extract(data, filename=url)
    elif ext == ".pdf":
        from app.modules.document_scanner import document_scanner
        return "PDF", [], document_scanner.scan_pdf_bytes(data, filename=name)
    elif ext in (".csv", ".xlsx", ".xlsm", ".xls"):
        from app.modules.extractors.spreadsheet_extractor import SpreadsheetExtractor
        modality, res = "CSV-XLSX", SpreadsheetExtractor().extract(data, filename=name)
    elif ext in IMAGE_EXT:
        from app.modules.extractors.image_extractor import ImageExtractor
        modality, res = "IMAGE", ImageExtractor().extract(data, filename=name)
    elif ext in (".html", ".htm"):
        from app.modules.extractors.web_extractor import WebExtractor
        modality, res = "WEB", WebExtractor().extract(data, filename=name)
    elif ext in CODE_EXT:
        from app.modules.extractors.code_extractor import CodeExtractor
        modality, res = "CODE", CodeExtractor().extract(data, filename=name)
    else:
        text = data.decode("utf-8", errors="replace")
        return "TEXT", [{"text": p, "location": f"paragraph {i + 1}", "hidden": False, "signals": []}
                        for i, p in enumerate(t for t in re.split(r"\n\s*\n", text) if t.strip())], None
    segs = []
    for s in res.segments:
        if s.content and s.content.strip():
            segs.append({"text": s.content.strip(), "location": s.location if isinstance(s.location, str) else json.dumps(s.location, default=str),
                         "hidden": bool(s.is_hidden), "signals": [str(getattr(t, "value", t)) for t in (s.threat_indicators or [])]})
    for w in res.extraction_warnings or []:
        segs.append({"text": "", "location": "warning", "hidden": False, "signals": [], "warning": w})
    return modality, segs, None


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def _score_one(text: str, role: str) -> Dict[str, Any]:
    """Calibrated risk: 100 x P(attack), with REVIEW / BLOCK thresholds chosen on validation data
    (app/core/calibrated_risk.py). Falls back to the legacy hand-set risk formula if no calibrator is installed."""
    patterns = scan_prompt(text)
    base = {"patterns": [m["id"] for m in patterns.get("matches", [])], "pattern_matches": patterns.get("matches", [])}
    if not text:
        return {**base, "p_inj": 0.0, "confidence": 1.0, "risk": 0, "category": "safe", "action": "PASS", "calibration": None}
    if calibrated_risk.available():
        # Same guards as the classifier path: non-prose fragments ("Summarize this", "Alice", "10") never reach
        # the detectors, and short document lines must look AI-targeted before the document detector counts.
        from app.modules.distilbert_classifier import AI_TARGETING_RE, is_non_prose
        sev = patterns.get("severity", 0)
        if not sev and is_non_prose(text)[0]:
            return {**base, "p_inj": 0.0, "confidence": 0.0, "risk": 0, "category": "safe", "action": "PASS",
                    "calibration": None, "skipped": "non_prose"}
        c = calibrated_risk.score(text, sev, role)
        if role == "document" and not sev and len(text.split()) < 25 and not AI_TARGETING_RE.search(text) \
                and c["signals"]["p_prompt_model"] < 0.5:
            c["probability_at_fit_rate"] = min(c["probability_at_fit_rate"], c["review_threshold"] - 1e-6)
            c["probability"] = min(c["probability"], 0.05)
            c["risk"] = min(c["risk"], 5)
        if role == "document":
            # documents: thresholds validated as calibrated probabilities (test: 96.6% / 1.4% review, 93.9% / 0.4% block)
            pf = c["probability_at_fit_rate"]
            action = "BLOCK" if pf >= c["block_threshold"] else "REWRITE" if pf >= c["review_threshold"] else "PASS"
        else:
            # prompts: the prompt detector decides; BLOCK needs a rule to agree (see _prompt_decision)
            p_model, sev = c["signals"]["p_prompt_model"], patterns.get("severity", 0)
            action = "BLOCK" if p_model >= 0.5 else ("REWRITE" if sev >= 50 else "PASS")
        return {**base, "p_inj": c["signals"]["p_prompt_model" if role == "prompt" else "p_document_model"],
                "confidence": c["probability"], "risk": c["risk"], "category": "attack" if action != "PASS" else "safe",
                "action": action, "calibration": c}
    cls = classify_prompt(text, role=role)
    scored = compute_risk_score(patterns["severity"], cls)
    return {**base, "p_inj": cls.get("model_p_inj"), "confidence": cls.get("confidence"), "risk": scored["risk_score"],
            "category": scored["category"], "action": scored["action"], "calibration": None}


def _score_text(text: str, role: str = "prompt") -> Dict[str, Any]:
    """Score the visible (sanitized) text AND everything the sanitizer strips as hidden (HTML comments,
    display:none, scripts, markdown link payloads). Hidden instructions are removed before forwarding, but they
    must still be detected, logged and explained; the riskiest of the parts decides."""
    clean = sanitize(text)
    best = {**_score_one(clean, role), "hidden_text": None}
    for frag in extract_hidden(text):
        s = _score_one(frag, role)
        if s["risk"] > best["risk"]:
            best = {**s, "hidden_text": frag}
    best["clean"] = clean
    return best


def _prompt_decision(s: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Map the risk engine action to the gateway decision for the user's own prompt."""
    if s["action"] == "BLOCK":
        # Auto-block only when a detection rule AND the model agree; one signal alone goes to human review.
        corroborated = bool(s["patterns"]) and (s["p_inj"] or 0) >= 0.5
        return ("BLOCK" if corroborated or policy() == "strict" else "REVIEW"), None
    if s["action"] == "REWRITE":
        # Never "clean and forward" a jailbreak: the whole message is the attack, so stripping the persona phrase
        # ("your dan now ...") would forward the harmful remainder. Same rule as the /analyze endpoint.
        if any(m.get("category") == "jailbreak" for m in s["pattern_matches"]):
            return "REVIEW", None
        if s["pattern_matches"]:
            from app.core.action_engine import _rewrite_prompt
            rewritten = _rewrite_prompt(s["clean"], s["pattern_matches"]).get("prompt", "")
            if len(re.findall(r"\b[A-Za-z]{2,}\b", rewritten)) >= 3:
                return ("REVIEW" if policy() == "strict" else "REWRITE"), rewritten
        return "REVIEW", None
    return "ALLOW", None


def _doc_decision(action: str, risk: int, calibrated: bool = False) -> str:
    if calibrated:  # thresholds already applied as validated probabilities in _score_one
        return {"BLOCK": "BLOCK", "REWRITE": "REVIEW", "REVIEW": "REVIEW"}.get(action, "ALLOW")
    if action == "BLOCK" or risk >= settings.block_risk_threshold:
        return "BLOCK"
    if action in ("REVIEW", "REWRITE") or risk >= settings.review_risk_threshold:
        return "REVIEW"
    return "ALLOW"


# ----------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------
def process(prompt: str, user_id: str = "user", files: Optional[List[Tuple[str, bytes]]] = None,
            urls: Optional[List[str]] = None) -> Dict[str, Any]:
    files, urls = files or [], [u for u in (urls or []) if u.strip()]
    timings, pipeline = {}, {}
    t = time.time()

    # 1. Content extraction
    sources = [(n, d, "file") for n, d in files] + [(u, u.encode("utf-8"), "url") for u in urls]
    extracted = []
    for name, data, kind in sources:
        try:
            modality, segs, pdf = _extract(name, data, kind)
            extracted.append({"name": name, "modality": modality, "segments": segs, "pdf": pdf, "error": None})
        except Exception as e:
            extracted.append({"name": name, "modality": "UNKNOWN", "segments": [], "pdf": None, "error": str(e)})
    pipeline["extraction"] = {"inputs": [{"name": "prompt", "modality": "TEXT", "segments": 1}] + [
        {"name": x["name"], "modality": x["modality"], "error": x["error"],
         "segments": (x["pdf"]["total_segments"] if x["pdf"] else len([s for s in x["segments"] if s.get("text")])),
         "hidden_segments": (x["pdf"]["cloaked_segments_count"] if x["pdf"] else sum(s["hidden"] for s in x["segments"]))}
        for x in extracted]}
    timings["extraction"] = time.time() - t

    # 2. Normalization
    t = time.time()
    from app.modules.normalizer import evaluate_normalized_evidence
    norm = evaluate_normalized_evidence(prompt)
    pipeline["normalization"] = {"modified": bool(norm.get("is_modified")), "anomalies": norm.get("anomaly_codes", []),
                                 "revealed_payload": norm.get("revealed_payload"),
                                 "normalized_preview": (norm.get("normalized_text") or "")[:300]}
    timings["normalization"] = time.time() - t

    # 3. Detection (user prompt + every extracted segment)
    t = time.time()
    p = _score_text(prompt)
    findings = []  # flagged content from attachments
    documents = []  # screened text that may be forwarded to the LLM
    for x in extracted:
        if x["error"]:
            findings.append({"source": x["name"], "modality": x["modality"], "location": "-", "text": x["error"],
                             "risk": 50, "decision": "REVIEW", "reason": "Could not extract this input safely", "hidden": False})
            continue
        if x["pdf"]:
            r = x["pdf"]
            for th in r.get("flagged_threats", []):
                ttext = th.get("full_text") or th.get("text_snippet") or ""
                # The PDF scanner decides (validated: 97.1% caught / 0.8% blocked on val PDFs); the shown risk is the
                # calibrated P(attack) of the flagged text when a calibrator is installed.
                shown = th.get("final_risk", 0)
                if calibrated_risk.available() and ttext:
                    shown = calibrated_risk.score(ttext, scan_prompt(ttext).get("severity", 0), "document")["risk"]
                findings.append({"source": x["name"], "modality": "PDF", "location": f"page {th.get('page', '-')}",
                                 "text": ttext[:400], "scanner_risk": th.get("final_risk", 0),
                                 "risk": shown, "decision": _doc_decision(th.get("action", ""), th.get("final_risk", 0)),
                                 "reason": th.get("reason", ""), "hidden": bool(th.get("is_hidden")),
                                 "signals": th.get("anomaly_codes", []), "p_inj": th.get("model_p_inj")})
            documents.append({"name": x["name"], "text": r.get("sanitized_document_text", "")})
            continue
        kept = []
        for sgm in x["segments"]:
            if not sgm.get("text"):
                continue
            s = _score_text(sgm["text"], role="document")
            calibrated = bool(s.get("calibration"))
            hidden_boost = 0 if calibrated else (15 if (sgm["hidden"] or sgm["signals"]) and s["risk"] > 15 else 0)
            risk = min(100, s["risk"] + hidden_boost)
            dec = _doc_decision(s["action"], risk, calibrated)
            if dec != "ALLOW":
                findings.append({"source": x["name"], "modality": x["modality"], "location": sgm["location"],
                                 "text": sgm["text"][:400], "risk": risk, "decision": dec, "hidden": sgm["hidden"],
                                 "signals": sgm["signals"] + s["patterns"], "p_inj": s["p_inj"],
                                 "reason": "Instruction aimed at the AI found inside " + x["modality"].lower() + " content"
                                           + (" (hidden)" if sgm["hidden"] else "")})
            else:
                kept.append(sgm["text"])
        documents.append({"name": x["name"], "text": "\n".join(kept)})
    pipeline["detection"] = {"prompt": {"risk": p["risk"], "patterns": p["patterns"], "p_inj": p["p_inj"],
                                        "hidden_text": p["hidden_text"]},
                             "attachment_findings": len(findings)}
    timings["detection"] = time.time() - t

    # 4. Classification (attack type)
    t = time.time()
    p_dec, rewritten = _prompt_decision(p)
    prompt_type = classify_attack_type(p["hidden_text"] or p["clean"], "user") if p_dec != "ALLOW" else {}
    for f in findings:
        f.update(classify_attack_type(f["text"], "document"))
    pipeline["classification"] = {"prompt": prompt_type or None,
                                  "attachments": [{"source": f["source"], "attack_type": f.get("attack_type"),
                                                   "technique": f.get("technique")} for f in findings]}
    timings["classification"] = time.time() - t

    # 5. Risk + 6. Action
    t = time.time()
    risk = max([p["risk"]] + [f["risk"] for f in findings])
    decision = max([p_dec] + [f["decision"] for f in findings], key=lambda d: SEVERITY[d])
    top = max(findings, key=lambda f: (SEVERITY[f["decision"]], f["risk"]), default=None)
    attack = prompt_type if SEVERITY[p_dec] >= SEVERITY[top["decision"] if top else "ALLOW"] and prompt_type else (
        {"attack_type": top["attack_type"], "technique": top["technique"], "type_confidence": top.get("type_confidence")} if top else {})
    forwarded = decision in ("ALLOW", "REWRITE") or policy() == "monitor"
    status = "forwarded" if forwarded else ("pending_review" if decision == "REVIEW" else "blocked")
    cal = p.get("calibration")
    pipeline["risk"] = {"score": risk, "calibrated": bool(cal),
                        "meaning": (f"P(attack) = {risk}%, calibrated on validation data and corrected to a "
                                    f"{round(100 * cal['assumed_attack_rate'])}% attack rate" if cal else "legacy hand-set scale"),

                        "signals": cal["signals"] if cal else None}
    pipeline["action"] = {"decision": decision, "status": status, "policy": policy()}
    timings["risk"] = timings["action"] = time.time() - t

    # 7. Explanation
    explanation = _explain(decision, p, p_dec, prompt_type, findings, rewritten)
    pipeline["explanation"] = {"text": explanation, "findings": findings}
    pipeline["timings_ms"] = {k: round(v * 1000, 1) for k, v in timings.items()}

    rec = {"id": uuid.uuid4().hex[:12], "created_at": _now(), "user_id": user_id, "prompt": prompt,
           "attachments": [{"name": x["name"], "modality": x["modality"]} for x in extracted],
           "decision": decision, "status": status, "risk_score": int(risk),
           "attack_type": attack.get("attack_type"), "technique": attack.get("technique"),
           "explanation": explanation, "pipeline": pipeline, "forwarded_prompt": rewritten or p["clean"],
           "documents": documents, "llm_response": None, "llm_meta": None}
    if forwarded:
        _forward(rec)
    save(rec)
    _audit(rec)
    return rec


def _explain(decision, p, p_dec, prompt_type, findings, rewritten) -> str:
    if decision == "ALLOW" and not findings:
        return "No attack found. The request was forwarded to the model."
    parts = []
    if p_dec != "ALLOW":
        kind = (prompt_type or {}).get("attack_type", "INJECTION").replace("_", " ").lower()
        where = (f"Your message hides an instruction in markup the reader would not see (\"{p['hidden_text'][:120]}\"); "
                 f"it is a {kind} attempt") if p.get("hidden_text") else f"Your message looks like a {kind} attempt"
        parts.append(f"{where} (risk {p['risk']}/100"
                     + (f"; matched rules: {', '.join(p['patterns'])}" if p["patterns"] else "") + ").")
        if rewritten:
            parts.append("The attack fragment was removed and the cleaned message was sent instead.")
    for f in sorted(findings, key=lambda f: -f["risk"])[:3]:
        parts.append(f"{f['source']} ({f['modality']}, {f['location']}): {f['reason']} - "
                     f"{f.get('technique', 'injection').replace('_', ' ').lower()} technique, risk {f['risk']}/100.")
    tail = {"BLOCK": "The request was blocked; the model never saw it.",
            "REVIEW": "The request is waiting for an administrator to verify it.",
            "REWRITE": "", "ALLOW": "The flagged content was removed before forwarding."}[decision]
    return " ".join(parts + ([tail] if tail else []))


def _forward(rec: Dict[str, Any]):
    docs = [d for d in rec.get("documents") or [] if d.get("text", "").strip()]
    result = complete(rec["forwarded_prompt"], docs)
    rec["llm_response"] = result["text"] if result["ok"] else None
    rec["llm_meta"] = {k: result[k] for k in ("ok", "provider", "model", "latency_ms", "error")}
    if rec["status"] in ("pending_review", "forwarded", "approved"):
        rec["status"] = "approved" if rec.get("reviewed_by") else "forwarded"
    if not result["ok"]:
        rec["status"] = "llm_error"


def _audit(rec: Dict[str, Any]):
    try:
        from app.core.init_db import log_audit
        log_audit(user_id=rec["user_id"], prompt=rec["prompt"][:2000], risk_score=rec["risk_score"],
                  attack_category=rec.get("attack_type") or "BENIGN", action_taken=rec["decision"],
                  detection_evidence=f"gateway_id={rec['id']} status={rec['status']}")
    except Exception as e:
        print(f"[audit] {e}")


def review(req_id: str, approve: bool, admin: str, note: str = "") -> Optional[Dict[str, Any]]:
    rec = get(req_id)
    if not rec:
        return None
    rec.update({"reviewed_by": admin, "review_note": note, "reviewed_at": _now()})
    if approve:
        rec["status"] = "approved"
        _forward(rec)
    else:
        rec["status"] = "rejected"
    save(rec)
    _audit({**rec, "decision": "APPROVED" if approve else "REJECTED"})
    return rec
