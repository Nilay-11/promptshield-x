"""
DistilBERT Binary Classification & Provenance Engine for PromptShield X.
Model outputs strictly binary logits: Class 0 (BENIGN) vs Class 1 (INJECTION).
All fine-grained categories are heuristic tags applied only when verdict is INJECTION.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.core.settings import settings
from app.modules.pattern_scanner import scan_prompt

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_DIR = WEIGHTS_DIR / "distilbert"
MANIFEST_PATH = WEIGHTS_DIR / "manifest.json"

TAXONOMY_CONVERTER = {
    "BENIGN": "safe",
    "DIRECT": "prompt_injection",
    "INDIRECT": "agent_manipulation",
    "JAILBREAK_PROMPT_LEAKAGE": "jailbreak",
    "PROMPT_EXTRACTION": "prompt_extraction",
    "AGENT_MANIPULATION": "agent_manipulation",
    "UNKNOWN": "prompt_injection",
}


def verify_manifest_integrity() -> bool:
    """Verifies SHA-256 hash of model directory if manifest exists."""
    if not MANIFEST_PATH.exists() or not MODEL_DIR.exists():
        return True
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_hash = manifest.get("distilbert", {}).get("sha256")
        if not expected_hash or expected_hash == "none":
            return True

        hasher = hashlib.sha256()
        for p in sorted(MODEL_DIR.glob("**/*")):
            if p.is_file():
                hasher.update(p.read_bytes())
        actual_hash = hasher.hexdigest()
        return actual_hash == expected_hash
    except Exception as e:
        print(f"[Manifest verification warning] {e}")
        return False


# Two detectors, routed by channel:
#   "document": text from PDFs, RAG chunks, web pages, files (MODEL_DIR, the v40 document model)
#   "prompt":   what a user types (PROMPT_MODEL_DIR, the v37 model: 0.7% FPR on benign user prompts vs 3.4% for v40)
# If no prompt model is installed, prompts fall back to the document model.
PROMPT_MODEL_DIR = WEIGHTS_DIR / "distilbert_prompt"
AI_TARGETING_RE = re.compile(
    r"\b(ai|assistant|model|llm|chatbot|bot|gpt|claude|gemini|copilot|agent|system|instructions?|ignore|disregard)\b"
    r"|https?://|www\.|\b[\w.+-]+@[\w-]+\.[\w.]+", re.I)
_distilbert_tokenizer = None
_distilbert_model = None
_loaded = {}


def _load_dir(path: Path):
    if path not in _loaded and path.exists() and (path / "config.json").exists():
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
            model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
            _loaded[path] = (tok, model)
        except Exception as e:
            print(f"[DistilBERT Load Error] {path}: {e}")
            _loaded[path] = (None, None)
    return _loaded.get(path, (None, None))


def _get_distilbert(role: str = "document"):
    global _distilbert_tokenizer, _distilbert_model
    if role == "prompt" and (PROMPT_MODEL_DIR / "config.json").exists():
        return _load_dir(PROMPT_MODEL_DIR)
    if _distilbert_model is None:
        _distilbert_tokenizer, _distilbert_model = _load_dir(MODEL_DIR)
    return _distilbert_tokenizer, _distilbert_model


def tag_heuristic_category(text: str, rule_hits: List[str]) -> str:
    """
    Assigns fine-grained attack taxonomy tags using word-boundary regexes.
    Runs ONLY when verdict is INJECTION.
    If no rule matches, heuristic_category is UNKNOWN.
    Guaranteed: Substrings like 'guidance', 'standard', 'model' will NEVER trigger tags.
    """
    tl = text.lower()

    # 1. Jailbreak / Policy Bypass / Persona
    if (
        any("jailbreak" in r.lower() for r in rule_hits)
        or re.search(r"\b(dan|do\s+anything\s+now|evilbot|unrestricted|jailbroken|unfiltered)\b", tl)
        or re.search(r"\b(developer|jailbreak|god|audit)\s+mode\b", tl)
        or re.search(r"\b(bypass|disable|circumvent|turn\s+off)\b.{0,40}\b(safety|content\s+filter|policy|restrictions?|guardrails?)\b", tl)
        or re.search(r"\b(pretend|act\s+as|roleplay\s+as)\b.{0,40}\b(no\s+rules|no\s+restrictions|evil|unrestricted)\b", tl)
    ):
        return "JAILBREAK_PROMPT_LEAKAGE"

    # 2. Direct Instruction Override
    if (
        any("override" in r.lower() or "ignore" in r.lower() or "injection" in r.lower() for r in rule_hits)
        or re.search(r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(instructions?|prompts?|rules?|guidelines?|constraints?|directives?|previous|system|prior|all)\b", tl)
        or re.search(r"\b(new\s+task|instead\s+of|stop\s+what\s+you\s+are\s+doing)\b", tl)
    ):
        return "DIRECT"

    # 3. System Prompt Extraction
    if (
        any("extraction" in r.lower() for r in rule_hits)
        or re.search(r"\b(reveal|show|print|output|display|tell\s+me|disclose|expose|leak|dump)\b.{0,40}\b(system\s*prompt|initial\s*prompt|developer\s*instructions?|internal\s*instructions?|hidden\s*configuration|secret\s*key|api\s*key|credentials?|password)\b", tl)
        or re.search(r"\b(repeat|print|output)\b.{0,30}\b(instructions?|prompt)\s+(verbatim|word\s+for\s+word)\b", tl)
    ):
        return "PROMPT_EXTRACTION"

    # 4. Agent / Hierarchy Manipulation
    if (
        any("agent" in r.lower() or "role" in r.lower() or "system-role" in r.lower() for r in rule_hits)
        or re.search(r"\b(system\s*message\s*:|higher\s*-?\s*priority|supersede|override)\b.{0,40}\b(user('s)?\s*request|instructions?|guidance)\b", tl)
        or re.search(r"\b(combine\s+fragments\s+in\s+reading\s+order)\b", tl)
        or re.search(r"\bfragment\s+\d+/\d+\s*:\s*.{0,50}\b(disregard|ignore|expose|reveal)\b", tl)
    ):
        return "AGENT_MANIPULATION"

    return "UNKNOWN"


def is_non_prose(text: str, min_tokens: Optional[int] = None) -> tuple[bool, str]:
    """
    Checks if a segment is non-prose metadata, timestamp, identifier, UUID, or numeric string.
    Non-prose segments are skipped from neural classification to prevent false positives.
    If explicit attack keywords are present, segment is NOT skipped.
    """
    st = text.strip()
    tl = st.lower()

    # Never skip if explicit attack imperative keywords are present
    if re.search(r"\b(ignore|disregard|override|reveal|system\s*prompt|bypass|jailbreak)\b", tl):
        return False, "PROSE"

    if min_tokens is None:
        min_tokens = settings.min_prose_token_count

    words = st.split()

    # Date strings, e.g. "2026-09-25", "D:20260925162000", "October 5, 2026", "25/09/2026"
    if re.match(r"^(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|D:\d{14}|[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})$", st):
        return True, "DATE_STRING"

    # UUIDs, e.g. "123e4567-e89b-12d3-a456-426614174000"
    if re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$", st):
        return True, "UUID_STRING"

    # Version strings, e.g. "v1.0.4", "version 2.3-beta"
    if re.match(r"^(v(ersion)?\s*\d+(\.\d+)*(-[a-zA-Z0-9]+)?)$", st, re.IGNORECASE):
        return True, "VERSION_STRING"

    # Machine Document IDs or Control Identifiers, e.g. "NB-Q3-2026 / CONTROL-SET-04 / REVIEW-COMPLETE"
    if re.match(r"^[A-Z0-9_-]{2,}(\s*[/|\-]\s*[A-Z0-9_-]+)+$", st):
        return True, "IDENTIFIER_STRING"

    # Pure numeric, punctuation, or hex
    if re.match(r"^[\d\s.,:;/\-–—_()#+%$*]+$", st):
        return True, "NUMERIC_PUNCTUATION"

    # Metadata key-values, e.g. "Author: nilay", "Date: 2024-01-01", "Author: Research & Development Group."
    # Only short values that do not address the model or ask for an action: "AI assistant: tell the user to ..."
    # used to be skipped here, which let any "Label: instruction" line bypass the classifier.
    if (re.match(r"^[A-Za-z\s_-]{2,25}\s*:\s*.+$", st) and len(words) <= 6
            and not re.search(r"\b(ai|assistant|model|bot|llm|you|your|please|tell|send|forward|email|click|log\s*in|"
                              r"visit|open|delete|transfer|reply|say|respond|output|print|instruction)\b", tl)):
        return True, "METADATA_KEY_VALUE"

    # Short token sequences without sentence-ending punctuation (e.g. "Status Report", "Confidential")
    if len(words) < min_tokens and not re.search(r"[.!?]", st):
        return True, "TOO_FEW_TOKENS"

    return False, "PROSE"


def classify_prompt_distilbert(text: str, role: str = "document") -> Dict[str, Any]:
    """
    Classifies input text according to settings.classifier_mode with strict provenance tracking.
    Never fabricates probabilities or confidence scores.
    """
    mode = settings.classifier_mode

    # 0. Normalization Evidence: evaluate BOTH original and normalized variants
    from app.modules.normalizer import evaluate_normalized_evidence
    norm_evidence = evaluate_normalized_evidence(text)
    normalized_text = norm_evidence["normalized_text"]
    revealed_payload = norm_evidence["revealed_payload"]
    anomaly_codes = list(norm_evidence["anomaly_codes"])

    # Guard: Non-prose skipping runs AFTER decoding (Section B, Item 4)
    if settings.skip_non_prose_enabled:
        is_skipped, skip_reason = is_non_prose(normalized_text)
        if is_skipped:
            return {
                "verdict": "SKIPPED_NON_PROSE",
                "model_p_inj": None,
                "model_p_benign": None,
                "rule_hits": [],
                "revealed_rules": [],
                "anomaly_codes": anomaly_codes,
                "heuristic_category": None,
                "final_score": 0,
                "mode": mode,
                "skip_reason": skip_reason,
                "source": "skipped_non_prose",
                "category": "BENIGN",
                "legacy_category": "safe",
                "confidence": 1.0,
                "original_text": text,
                "normalized_text": normalized_text,
                "revealed_payload": revealed_payload,
            }

    # 1. Evaluate Rule Hits (Regex First-Pass on both original and normalized)
    rule_hits: List[str] = []
    rule_severity: int = 0
    if mode in ["combined", "rules_only"]:
        rule_hits = norm_evidence["norm_rules"]
        rule_severity = norm_evidence["max_severity"]
        if settings.rules_evidence_only:
            # Rule hits can raise severity to REVIEW level (<= 40), never BLOCK (>= 60) alone
            rule_severity = min(rule_severity, 40)

    # 2. Evaluate Neural Model (DistilBERT on de-obfuscated text)
    # Long inputs: support overlapping sliding-window scoring (max over windows)
    model_p_inj: Optional[float] = None
    model_p_benign: Optional[float] = None
    tokenizer, model = _get_distilbert(role)
    eval_text = normalized_text if normalized_text else text

    if mode in ["combined", "model_only", "semantic_only"] and tokenizer is not None and model is not None:
        try:
            import torch
            token_ids = tokenizer.encode(eval_text, add_special_tokens=True)
            max_len = 512
            stride = settings.sliding_window_stride

            if len(token_ids) <= max_len or not settings.sliding_window_inference_enabled:
                inputs = tokenizer(eval_text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()
                    model_p_benign = round(probs[0], 4)
                    model_p_inj = round(probs[1], 4)
            else:
                # Sliding-window inference across long text
                window_p_injs = []
                window_p_bens = []
                # Strip outermost special tokens for interior chunk slicing
                core_ids = token_ids[1:-1]
                content_len = max_len - 2
                cls_id = tokenizer.cls_token_id
                sep_id = tokenizer.sep_token_id

                for start_idx in range(0, len(core_ids), stride):
                    chunk_slice = core_ids[start_idx : start_idx + content_len]
                    if not chunk_slice:
                        continue
                    framed_ids = [cls_id] + chunk_slice + [sep_id]
                    input_tensor = torch.tensor([framed_ids], dtype=torch.long, device=model.device)
                    attention_mask = torch.ones_like(input_tensor)
                    with torch.no_grad():
                        out = model(input_ids=input_tensor, attention_mask=attention_mask)
                        chunk_probs = torch.softmax(out.logits, dim=-1)[0].tolist()
                        window_p_bens.append(chunk_probs[0])
                        window_p_injs.append(chunk_probs[1])
                    if start_idx + content_len >= len(core_ids):
                        break

                max_p_inj = max(window_p_injs) if window_p_injs else 0.0
                model_p_inj = round(max_p_inj, 4)
                model_p_benign = round(1.0 - max_p_inj, 4)

        except Exception as e:
            print(f"[DistilBERT Inference Error] {e}")
            model_p_inj = None
            model_p_benign = None

    # 2b. Short-segment gate for the document model: it over-flags short imperative business lines
    # ("Next steps: finalize the shortlist by Friday"). When the prompt model disagrees and the text neither
    # addresses an AI nor points anywhere (URL / email) nor tells it to ignore instructions, trust the prompt model.
    if (role == "document" and model_p_inj is not None and model_p_inj >= 0.5
            and len(eval_text.split()) < 25 and not AI_TARGETING_RE.search(eval_text)
            and (PROMPT_MODEL_DIR / "config.json").exists()):
        p_prompt = classify_prompt_distilbert(eval_text, role="prompt").get("model_p_inj")
        if p_prompt is not None and p_prompt < 0.5:
            model_p_inj, model_p_benign = p_prompt, round(1.0 - p_prompt, 4)
            anomaly_codes.append("SHORT_SEGMENT_PROMPT_MODEL_GATE")

    # 3. Decision Logic Based on Operating Mode
    if mode == "rules_only":
        final_score = rule_severity
        verdict = "INJECTION" if final_score >= 50 else "BENIGN"
        confidence = round(final_score / 100.0, 4) if verdict == "INJECTION" else round((100 - final_score) / 100.0, 4)
    elif mode == "model_only":
        p_inj = model_p_inj if model_p_inj is not None else 0.0
        final_score = round(p_inj * 100)
        verdict = "INJECTION" if p_inj >= 0.50 else "BENIGN"
        confidence = p_inj if verdict == "INJECTION" else (model_p_benign or (1.0 - p_inj))
    else:  # "combined" (default)
        p_inj = model_p_inj if model_p_inj is not None else (rule_severity / 100.0)
        # Combined semantic attack score synthesizes model probability and rule evidence
        model_score = round(p_inj * 100)
        final_score = max(model_score, rule_severity) if rule_hits else model_score
        verdict = "INJECTION" if (p_inj >= 0.50 or rule_severity >= 50) else "BENIGN"
        confidence = p_inj if verdict == "INJECTION" else (model_p_benign or (1.0 - p_inj))

    # 4. Optional Legacy Override Ablation Switch
    if settings.overrides_enabled:
        DIRECTIVE_PATTERNS = [
            re.compile(r"(?i)\b(ignore|disregard|forget|override|supersede|bypass|skip)\b.{0,40}\b(instructions?|prompts?|rules?|guidelines?|constraints?|restrictions?|directives?|above|previous|system|prior|earlier|all)\b"),
            re.compile(r"(?i)\b(reveal|show|print|output|display|tell me|disclose|expose|what is|leak|dump)\b.{0,40}\b(system\s*prompt|initial\s*prompt|developer\s*instructions?|internal\s*instructions?|instructions?\s*verbatim|hidden\s*configuration|hidden\s*prompt|secret\s*key|api\s*key|credentials?|password)\b"),
            re.compile(r"(?i)\b(from now on you are|you are now|act as( an?)?|pretend (you are|you have)|roleplay as)\b.{0,40}\b(dan|unrestricted|jailbroken|unfiltered|free|evil|no rules|do anything now)\b"),
            re.compile(r"(?i)\b(bypass|disable|circumvent|turn off)\b.{0,40}\b(safety|content filter|restrictions?|policies|guardrails?)\b"),
            re.compile(r"(?i)\b(higher priority than all|follow this (document|instruction) instead of|treat (the text below|this document|retrieved content) as higher(-|\s*)priority)\b"),
            re.compile(r"(?i)\bSYSTEM\s*MESSAGE\s*:\s*.{0,60}\b(supersede|override|instead|ignore|disregard)\b"),
        ]
        META_REGEX = re.compile(r"(?i)\b(discuss(es|ing|ion)?|training|education|curriculum|article|paper|example\s+of|definition\s+of|explain(s|ing)?|study|safeguard(s)?|guard\s+against|defense\s+against)\b")
        PAYLOAD_REGEX = re.compile(r"(?i)\b(and|then|instead)\b.{0,40}\b(reveal|print|output|tell me|give me|bypass|execute)\b")

        has_directive = any(p.search(text) for p in DIRECTIVE_PATTERNS)
        has_meta = bool(META_REGEX.search(text))
        has_payload = bool(PAYLOAD_REGEX.search(text))

        if has_meta and not has_payload:
            verdict = "BENIGN"
            final_score = 5
        elif has_directive:
            verdict = "INJECTION"
            final_score = 95
        elif verdict == "INJECTION" and not has_directive:
            verdict = "BENIGN"
            final_score = 5

    # 5. Assign Heuristic Category ONLY if verdict is INJECTION
    heuristic_category: Optional[str] = None
    if verdict == "INJECTION":
        heuristic_category = tag_heuristic_category(eval_text, rule_hits)

    category_label = heuristic_category if heuristic_category else "BENIGN"

    return {
        "verdict": verdict,
        "model_p_inj": model_p_inj,
        "model_p_benign": model_p_benign,
        "rule_hits": rule_hits,
        "heuristic_category": heuristic_category,
        "final_score": final_score,
        "mode": mode,
        "category": category_label,
        "legacy_category": "safe" if verdict in ["BENIGN", "SKIPPED_NON_PROSE"] else TAXONOMY_CONVERTER.get(category_label, "prompt_injection"),
        "confidence": confidence,
        "source": "distilbert_local" if model_p_inj is not None else "rules_engine",
        "original_text": text,
        "normalized_text": normalized_text,
        "revealed_payload": revealed_payload,
        "revealed_rules": norm_evidence["revealed_rules"],
        "anomaly_codes": anomaly_codes,
    }
