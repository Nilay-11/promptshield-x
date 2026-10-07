"""
API routes for PromptShield X.

Pipeline: sanitizer (6.1) -> pattern scanner (6.2) -> semantic classifier
(6.3) -> risk engine (6.9) -> action engine (6.10).

/analyze-rag (Person B — minimal RAG path): runs the SAME per-item pipeline
(sanitizer -> pattern scanner -> semantic classifier -> risk engine) once per
retrieved chunk, instead of the prompt as a whole. No new detection model —
chunk_scanner / anomaly_detector / reliability_filter / evaluator_llm
(6.4-6.8) are explicitly out of scope for this pass; overall verdict is
currently just "riskiest chunk wins," which is a simplification standing in
for the reliability/consensus filter documented as future work.

NOTE: semantic_classifier is a zero-shot HF pipeline (~0.5-1.5s/call on CPU).
/analyze-rag latency scales linearly with chunk count as a result.
"""

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse

import json
import os
from app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeRagResponse,
    ChunkRiskResult,
)
from app.modules.sanitizer import extract_hidden, sanitize
from app.modules.pattern_scanner import scan_prompt
from app.modules.semantic_classifier import classify_prompt
from app.core.risk_engine import compute_risk_score
from app.core.action_engine import apply_action
from app.core.init_db import log_audit
from app.modules.attack_type import classify_attack_type

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_prompt(payload: AnalyzeRequest):
    clean_prompt = sanitize(payload.prompt)
    # scan hidden markup the sanitizer strips (HTML comments, display:none ...) as well as the visible text
    scan_text = " ".join([clean_prompt] + extract_hidden(payload.prompt))

    pattern_result = scan_prompt(scan_text)
    classification = classify_prompt(scan_text)
    scored = compute_risk_score(pattern_result["severity"], classification)

    action = scored["action"]
    # If action is BLOCK but pattern matches are removable and surviving prompt has legitimate intent (never for jailbreaks):
    if action == "BLOCK" and scored.get("category") in ["prompt_injection", "DIRECT", "prompt_extraction"] and pattern_result.get("matches"):
        from app.core.action_engine import _rewrite_prompt
        import re
        rewritten_dict = _rewrite_prompt(clean_prompt, pattern_result["matches"])
        remaining_words = [w for w in re.findall(r"\b[A-Za-z]{2,}\b", rewritten_dict.get("prompt", "")) if w.lower() not in {"and", "or", "the", "a", "an", "then", "to", "is"}]
        if len(remaining_words) >= 2 and rewritten_dict.get("removed_fragments"):
            action = "REWRITE"
            scored["action"] = "REWRITE"

    outcome = apply_action(action, clean_prompt, pattern_result["matches"])

    details = (
        f"pattern_matches={[m['id'] for m in pattern_result['matches']]}, "
        f"classifier_confidence={classification['confidence']}, "
        f"removed_fragments={outcome['removed_fragments']}"
    )

    try:
        log_audit(
            user_id=payload.user_id,
            prompt=payload.prompt,
            risk_score=scored["risk_score"],
            attack_category=scored["category"],
            action_taken=scored["action"],
            detection_evidence=details,
        )
    except Exception as e:
        print(f"Failed to log audit entry: {e}")

    typed = classify_attack_type(clean_prompt, "user") if scored["action"] != "PASS" else {}
    return AnalyzeResponse(
        action=scored["action"],
        risk_score=scored["risk_score"],
        category=scored["category"],
        details=details,
        rewritten_prompt=outcome["prompt"] if scored["action"] == "REWRITE" else None,
        **typed,
    )


@router.post("/analyze-rag", response_model=AnalyzeRagResponse)
def analyze_rag_context(payload: AnalyzeRequest):
    """
    Indirect (RAG) prompt injection check — minimal path.
    Reuses sanitizer + pattern_scanner + semantic_classifier + risk_engine
    per chunk. Full version (6.4-6.8: chunk scanner, anomaly detector,
    reliability filter, evaluator LLM) is documented as future work.
    """
    chunks = payload.retrieved_chunks or []

    chunk_results: list[ChunkRiskResult] = []
    for i, chunk in enumerate(chunks):
        clean_chunk = sanitize(chunk)
        scan_text = " ".join([clean_chunk] + extract_hidden(chunk))

        pattern_result = scan_prompt(scan_text)
        classification = classify_prompt(scan_text, role="document")
        scored = compute_risk_score(pattern_result["severity"], classification)

        chunk_action = scored["action"]
        if chunk_action == "REWRITE":
            from app.core.action_engine import _rewrite_prompt
            import re
            rewritten_dict = _rewrite_prompt(clean_chunk, pattern_result.get("matches", []))
            remaining_words = [w for w in re.findall(r"\b[A-Za-z]{2,}\b", rewritten_dict.get("prompt", "")) if w.lower() not in {"and", "or", "the", "a", "an", "then", "to", "is"}]
            if len(remaining_words) < 2:
                chunk_action = "BLOCK"

        chunk_results.append(
            ChunkRiskResult(
                index=i,
                chunk_preview=(clean_chunk[:120] + "...") if len(clean_chunk) > 120 else clean_chunk,
                risk_score=scored["risk_score"],
                category=scored["category"],
                action=chunk_action,
                pattern_matches=[m["id"] for m in pattern_result["matches"]],
                classifier_confidence=classification["confidence"],
                **(classify_attack_type(clean_chunk, "rag") if chunk_action != "PASS" else {}),
            )
        )

    if chunk_results:
        if any(c.action == "BLOCK" for c in chunk_results):
            overall_action = "BLOCK"
        elif any(c.action == "REWRITE" for c in chunk_results):
            overall_action = "REWRITE"
        else:
            overall_action = "PASS"
        riskiest = max(chunk_results, key=lambda c: c.risk_score)
        overall_risk_score = riskiest.risk_score
    else:
        overall_risk_score = 0
        overall_action = "PASS"

    return AnalyzeRagResponse(
        total_chunks=len(chunk_results),
        overall_risk_score=overall_risk_score,
        overall_action=overall_action,
        chunks=chunk_results,
    )


@router.post("/analyze-pdf")
async def analyze_pdf_upload(file: UploadFile = File(...)):
    """
    Deeply inspects an uploaded PDF document:
    - Extracts body text, metadata, form fields, and annotations
    - Detects microscopic/zero-point fonts (< 2.0pt), off-canvas text, and invisible white text
    - Evaluates all segments through the multi-layer security engine (DistilBERT + Regex + Anomaly)
    - Returns structured page-by-page risk assessment and sanitized text
    """
    from app.modules.document_scanner import document_scanner
    
    try:
        pdf_bytes = await file.read()
        if not pdf_bytes:
            return {"status": "error", "message": "Uploaded file is empty."}
            
        result = document_scanner.scan_pdf_bytes(pdf_bytes, filename=file.filename or "document.pdf")
        
        # Log audit entry for document analysis
        try:
            log_audit(
                user_id="pdf_inspector",
                prompt=f"[PDF SCAN] {file.filename} ({result['total_pages']} pages, {result['total_segments']} segments)",
                risk_score=result["overall_risk_score"],
                attack_category=result.get("primary_heuristic_category", "BENIGN") if result["flagged_threats_count"] > 0 else "BENIGN",
                action_taken=result["overall_action"],
                detection_evidence=f"cloaked_segments={result['cloaked_segments_count']}, flagged_threats={result['flagged_threats_count']}"
            )
        except Exception as e:
            print(f"[Audit Log Warning] {e}")
            
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": f"PDF scanning error: {str(e)}"}


@router.get("/api/test-suite/16-families")
def get_16_families_test_suite():
    """Returns the 16 attack families and matching hard benign negative benchmark cases."""
    from app.core.taxonomy import BENCHMARK_CASES
    
    cases_data = []
    for c in BENCHMARK_CASES:
        cases_data.append({
            "id": c["id"],
            "family": c["family"].value,
            "modality": c["modality"].value,
            "expected_label": c["expected_label"].value,
            "attack_prompt": c["attack"],
            "benign_prompt": c["benign"]
        })
    return {"status": "ok", "cases": cases_data}


@router.post("/api/test-suite/run-case")
def run_single_benchmark_case(payload: dict):
    """Executes a single test case prompt through the live firewall pipeline."""
    prompt = payload.get("prompt", "")
    if not prompt:
        return {"status": "error", "message": "Prompt is required."}

    from app.modules.distilbert_classifier import classify_prompt_distilbert
    clean = sanitize(prompt)
    patterns = scan_prompt(clean)
    distil = classify_prompt_distilbert(clean)
    scored = compute_risk_score(patterns["severity"], distil)

    return {
        "status": "ok",
        "result": {
            "action": scored["action"],
            "risk_score": scored["risk_score"],
            "distil_category": distil["category"],
            "distil_confidence": distil["confidence"],
            "is_blocked": scored["action"] in ["BLOCK", "REWRITE"]
        }
    }



@router.get("/admin/logs")
def get_audit_logs(limit: int = 100):
    """Backs the audit dashboard (Chapter 6.12)."""
    import sqlite3
    from app.core.init_db import DB_PATH
    
    if not DB_PATH.exists():
        return {"status": "ok", "logs": []}
        
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, timestamp, user_id, prompt, risk_score, attack_category, action_taken, detection_evidence
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()
        
        logs = []
        for row in rows:
            logs.append({
                "id": row[0],
                "timestamp": row[1],
                "user_id": row[2],
                "prompt": row[3],
                "risk_score": row[4],
                "attack_category": row[5],
                "action_taken": row[6],
                "detection_evidence": row[7]
            })
        return {"status": "ok", "logs": logs}
    except Exception as e:
        return {"status": "error", "message": str(e), "logs": []}
    finally:
        conn.close()


@router.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serves the bespoke PromptShield X Dashboard."""
    template_path = os.path.join("dashboard", "templates", "dashboard.html")
    if not os.path.exists(template_path):
        return HTMLResponse("<h1>Dashboard HTML template not found.</h1>", status_code=404)
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@router.get("/demo", response_class=HTMLResponse)
def get_demo():
    """Demo page: upload your own PDF, test prompts / RAG chunks, and view unseen-data results."""
    template_path = os.path.join("dashboard", "templates", "demo.html")
    if not os.path.exists(template_path):
        return HTMLResponse("<h1>Demo template not found.</h1>", status_code=404)
    with open(template_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# Final tests run ONCE on sources never used for training or tuning (see eval/DATASETS.md).
# Each entry: (results file, scorer key, slice, operating point, label shown in the demo).
_UNSEEN_RESULTS = [
    ("v37_eval_final.json", "distilbert:models/distilbert_v37", "ALL", "at_val_fpr1",
     "AgentDojo agent tool outputs + PubMed abstracts", "v37 (app model)"),
    ("v38_eval_final.json", "distilbert:models/distilbert_v37", "ALL", "at_val_fpr1",
     "SaTML CTF attacks hidden in US government reports", "v37 (app model)"),
    ("v40_eval_final.json", "distilbert:models/distilbert_v40", "source:rag_chunk_attacked_gentel_cnn_final", "at_val_fpr5",
     "GenTelBench attacks hidden in CNN/DailyMail RAG chunks", "v40 (RAG research model)"),
]


@router.get("/api/eval-summary")
def eval_summary():
    """Recall / false-positive rate of the one-time final tests on unseen data, read from eval/results."""
    out = []
    for fname, scorer, sl, op, label, model in _UNSEEN_RESULTS:
        path = os.path.join("eval", "results", fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                slices = json.load(f)["scorers"][scorer]["slices"]
            rec = slices[sl][op].get("recall")
            # FPR: the benign rows of the same test (for chunk slices use the matching clean chunks)
            fpr_slice = sl.replace("attacked", "clean") if "attacked" in sl else sl
            fpr = slices.get(fpr_slice, {}).get(op, {}).get("fpr")
            out.append({"test": label, "model": model, "recall": rec, "fpr": fpr,
                        "auroc": slices["ALL"].get("auroc")})
        except Exception as e:
            out.append({"test": label, "model": model, "error": str(e)})
    pipe = os.path.join("eval", "results", "format_suite_v38_final_results.json")
    if os.path.exists(pipe):
        with open(pipe, "r", encoding="utf-8") as f:
            r = json.load(f)["scorers"]["production_document_scanner"]["results"]["val_fpr5"]["ALL"]
        out.append({"test": "Full app pipeline on rendered PDFs/HTML (SaTML in GAO reports)",
                    "model": "v37 + PDF forensics", "recall": r.get("recall"), "fpr": r.get("fpr"), "auroc": None})
    return {"status": "ok", "results": out,
            "note": "Each final test was run once, on sources never used for training or threshold tuning."}
