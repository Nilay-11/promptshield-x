"""
PromptShield X - Document & PDF Security Scanner.

Inspects incoming files (PDFs, text, RAG payloads), extracts visible and cloaked content,
evaluates every segment against the multi-layer security engine (DistilBERT + Regex + Anomaly),
and returns structured forensic evidence.
"""

import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.extractors.pdf_extractor import PDFExtractor
from app.modules.sanitizer import sanitize
from app.modules.pattern_scanner import scan_prompt
from app.modules.distilbert_classifier import classify_prompt_distilbert
from app.modules.anomaly_detector import detect_anomaly
from app.core.risk_engine import compute_risk_score
from app.core.settings import settings


class DocumentScanner:
    def __init__(self):
        self.pdf_extractor = PDFExtractor(min_visible_font_size=settings.min_visible_font_size)

    def scan_pdf_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """
        Extracts and deeply scans a PDF document for direct and cloaked indirect prompt injections.
        Implements calibrated multiplicative risk scoring:
            risk = semantic * (1 + structural_amplifier)
        Structural features can only amplify semantic risk, never create it from zero.
        """
        extracted = self.pdf_extractor.extract(pdf_bytes, filename=filename)
        
        segments_results: List[Dict[str, Any]] = []
        flagged_threats: List[Dict[str, Any]] = []
        page_risks: Dict[int, List[int]] = {}
        clean_text_segments: List[str] = []

        max_risk = 0

        for idx, seg in enumerate(extracted.segments):
            raw_text = seg.content.strip()
            if not raw_text:
                continue

            clean_text = sanitize(raw_text)
            pattern_res = scan_prompt(clean_text)
            distil_res = classify_prompt_distilbert(clean_text)

            # ------------------------------------------------------------------
            # 1. Semantic Attack Score (0-100) & Provenance (Phase 1, Req 1 & 2)
            # ------------------------------------------------------------------
            verdict = distil_res.get("verdict", "BENIGN")
            model_p_inj = distil_res.get("model_p_inj")
            model_p_benign = distil_res.get("model_p_benign")
            rule_hits = distil_res.get("rule_hits", [])
            heuristic_cat = distil_res.get("heuristic_category")
            classifier_mode = distil_res.get("mode", settings.classifier_mode)

            anomaly_codes = list(seg.metadata.get("anomaly_codes", []))
            if verdict == "SKIPPED_NON_PROSE":
                if "SKIPPED_NON_PROSE" not in anomaly_codes:
                    anomaly_codes.append("SKIPPED_NON_PROSE")
                semantic_attack_score = 0
            elif settings.classifier_mode == "model_only":
                semantic_attack_score = round((model_p_inj or 0.0) * 100)
            elif settings.classifier_mode == "rules_only":
                semantic_attack_score = pattern_res.get("severity", 0) if pattern_res.get("matches") else 0
            else:
                # Combined mode
                semantic_attack_score = distil_res.get("final_score", 0)

            # ------------------------------------------------------------------
            # 2. Structural Anomaly Features
            # ------------------------------------------------------------------
            layout_anomaly_score = seg.layout_anomaly_score
            visibility_score = seg.visibility_score
            position_anomaly_score = seg.position_anomaly_score
            metadata_anomaly_score = seg.metadata_anomaly_score

            max_struct = max(layout_anomaly_score, visibility_score, position_anomaly_score, metadata_anomaly_score)
            structural_amplifier = (max_struct / 100.0) * settings.structural_amplifier_max

            # ------------------------------------------------------------------
            # 3. Multiplicative Risk Engine & Configurable Tiers (Phase 1.5 / Phase 3.5)
            # ------------------------------------------------------------------
            if settings.classifier_mode == "structural_only":
                # D.3: Structural-only mode derives risk strictly from layout anomalies, action is REVIEW at most (>= 30), never BLOCK
                final_risk = int(max_struct)
                action = "REVIEW" if final_risk >= settings.review_risk_threshold else "PASS"
            elif settings.additive_hidden_penalty_enabled:
                # Legacy additive mode for ablation
                raw_combined = round(
                    0.60 * semantic_attack_score
                    + 0.20 * layout_anomaly_score
                    + 0.10 * visibility_score
                    + 0.05 * position_anomaly_score
                    + 0.05 * metadata_anomaly_score
                )
                if seg.is_hidden or seg.cloaking_signal != "NONE":
                    final_risk = max(88, min(100, raw_combined + 30))
                else:
                    final_risk = max(70, min(100, raw_combined)) if semantic_attack_score > 60 else min(60, raw_combined)
                action = "BLOCK" if final_risk >= settings.block_risk_threshold else ("REVIEW" if final_risk >= settings.review_risk_threshold else "PASS")
            else:
                # Multiplicative risk formula: risk = semantic * (1 + structural_amplifier)
                # Structural features can only amplify semantic risk, never create it from zero!
                if semantic_attack_score == 0:
                    final_risk = 0
                    action = "PASS"
                else:
                    multiplied = round(semantic_attack_score * (1.0 + structural_amplifier))
                    final_risk = min(100, max(0, multiplied))

                    # Configurable Tiers:
                    # Tier 1: Low semantic tier (semantic <= settings.semantic_tier_low)
                    # Structural features cannot raise action above REVIEW, regardless of magnitude.
                    if semantic_attack_score <= settings.semantic_tier_low:
                        final_risk = min(final_risk, settings.block_risk_threshold - 1)
                        if final_risk >= settings.review_risk_threshold:
                            action = "REVIEW"
                        else:
                            action = "PASS"

                    # Tier 2: Ambiguous tier (semantic_tier_low < semantic < settings.semantic_tier_high)
                    # By default (ambiguous_tier_can_block=False), raises to REVIEW at most, never BLOCK.
                    elif semantic_attack_score < settings.semantic_tier_high:
                        if settings.ambiguous_tier_can_block:
                            if final_risk >= settings.block_risk_threshold:
                                action = "BLOCK"
                            elif final_risk >= settings.review_risk_threshold:
                                action = "REVIEW"
                            else:
                                action = "PASS"
                        else:
                            final_risk = min(final_risk, settings.block_risk_threshold - 1)
                            if final_risk >= settings.review_risk_threshold:
                                action = "REVIEW"
                            else:
                                action = "PASS"

                    # Tier 3: High semantic tier (semantic >= settings.semantic_tier_high)
                    # BLOCK regardless of structure
                    else:
                        final_risk = max(final_risk, settings.block_risk_threshold)
                        action = "BLOCK"

                # D.2: Hidden-text REVIEW path
                # If hidden anomaly is present and (model_p_inj >= hidden_text_review_p_threshold or rule_hits),
                # action must be at least REVIEW. Never BLOCK on structure alone.
                has_hidden_anomaly = bool(seg.is_hidden or seg.cloaking_signal != "NONE" or max_struct >= 30)
                p_inj_val = model_p_inj if model_p_inj is not None else 0.0
                if has_hidden_anomaly and (p_inj_val >= settings.hidden_text_review_p_threshold or bool(rule_hits)):
                    if action == "PASS":
                        action = "REVIEW"
                        final_risk = max(final_risk, settings.review_risk_threshold)

            if final_risk > max_risk:
                max_risk = final_risk

            for ac in distil_res.get("anomaly_codes", []):
                if ac not in anomaly_codes:
                    anomaly_codes.append(ac)

            # ------------------------------------------------------------------
            # 4. Explicit Audit Reason Codes
            # ------------------------------------------------------------------
            if verdict == "SKIPPED_NON_PROSE":
                reason = f"Non-prose segment ({distil_res.get('skip_reason', 'SKIPPED')}) bypassed neural classification"
            elif action == "BLOCK":
                if anomaly_codes:
                    reason = f"Adversarial instruction detected with layout anomalies ({', '.join(anomaly_codes)})"
                else:
                    reason = f"Explicit semantic injection directive ({heuristic_cat or 'DIRECT'})"
            elif action == "REVIEW":
                if anomaly_codes:
                    reason = f"Suspicious content with layout anomalies ({', '.join(anomaly_codes)}); flagged for review"
                else:
                    reason = "Ambiguous prompt content; flagged for review"
            else:  # PASS
                if anomaly_codes:
                    reason = f"Benign segment with layout anomalies logged ({', '.join(anomaly_codes)}); no malicious directive"
                else:
                    reason = "Standard benign document content"

            page_num = seg.location.get("page", 1) if isinstance(seg.location, dict) else 1
            if page_num not in page_risks:
                page_risks[page_num] = []
            page_risks[page_num].append(final_risk)

            is_malicious = (action in ["BLOCK", "REVIEW"] and semantic_attack_score > 0)

            segment_data = {
                "index": idx + 1,
                "page": page_num,
                "location": seg.location,
                "text_snippet": (clean_text[:140] + "...") if len(clean_text) > 140 else clean_text,
                "full_text": clean_text,
                "extracted_text": clean_text,
                "segment_type": seg.segment_type,
                "cloaking_signal": seg.cloaking_signal,
                "is_hidden": seg.is_hidden,
                "layout_anomaly_score": int(layout_anomaly_score),
                "visibility_score": int(visibility_score),
                "position_anomaly_score": int(position_anomaly_score),
                "metadata_anomaly_score": int(metadata_anomaly_score),
                "semantic_attack_score": int(semantic_attack_score),
                "layout_risk": int(max_struct),
                "semantic_risk": int(semantic_attack_score),
                "final_risk": final_risk,
                "risk_score": final_risk,
                "distilbert_class": heuristic_cat if heuristic_cat else verdict,
                "semantic_class": verdict,
                "heuristic_category": heuristic_cat,
                "model_p_inj": model_p_inj,
                "model_p_benign": model_p_benign,
                "rule_hits": rule_hits,
                "mode": classifier_mode,
                "anomaly_codes": anomaly_codes,
                "distilbert_confidence": model_p_inj if model_p_inj is not None else float(distil_res.get("confidence", 0.0)),
                "pattern_matches": rule_hits,
                "threat_indicators": [t.value for t in seg.threat_indicators],
                "action": action,
                "reason": reason,
                "is_malicious": is_malicious,
                "metadata": seg.metadata,
            }

            segments_results.append(segment_data)

            if is_malicious:
                flagged_threats.append(segment_data)
            else:
                clean_text_segments.append(clean_text)

        # ------------------------------------------------------------------
        # 5. Sliding-Window Pass Over Reading Order (Section 3, Item 3)
        # Catches payloads split across short or skipped segments
        # ------------------------------------------------------------------
        window_triggers = []
        if settings.reading_order_window_enabled and len(segments_results) >= 2:
            w_size = settings.reading_order_window_size
            for i in range(len(segments_results)):
                window_slice = segments_results[i : min(i + w_size, len(segments_results))]
                if len(window_slice) < 2:
                    continue
                w_text = " ".join(s["full_text"] for s in window_slice).strip()
                w_clean = sanitize(w_text)
                w_pattern_res = scan_prompt(w_clean)
                w_distil_res = classify_prompt_distilbert(w_clean)

                w_semantic = w_distil_res.get("final_score", 0)
                w_max_struct = max(s["layout_risk"] for s in window_slice)
                w_amp = (w_max_struct / 100.0) * settings.structural_amplifier_max
                w_multiplied = round(w_semantic * (1.0 + w_amp)) if w_semantic > 0 else 0
                w_final_risk = min(100, max(0, w_multiplied))

                if w_semantic <= settings.semantic_tier_low:
                    w_final_risk = min(w_final_risk, settings.block_risk_threshold - 1)
                    w_action = "REVIEW" if w_final_risk >= settings.review_risk_threshold else "PASS"
                elif w_semantic < settings.semantic_tier_high:
                    if settings.ambiguous_tier_can_block:
                        w_action = "BLOCK" if w_final_risk >= settings.block_risk_threshold else ("REVIEW" if w_final_risk >= settings.review_risk_threshold else "PASS")
                    else:
                        w_final_risk = min(w_final_risk, settings.block_risk_threshold - 1)
                        w_action = "REVIEW" if w_final_risk >= settings.review_risk_threshold else "PASS"
                else:
                    w_final_risk = max(w_final_risk, settings.block_risk_threshold)
                    w_action = "BLOCK"

                w_has_attack = (
                    w_distil_res.get("verdict") == "INJECTION"
                    or bool(w_pattern_res.get("rule_hits"))
                    or w_action == "BLOCK"
                )
                if w_has_attack and w_action in ["BLOCK", "REVIEW"] and w_semantic > 0:
                    trigger_label = f"segments {i + 1}-{i + len(window_slice)}"
                    w_threat = {
                        "window_type": "READING_ORDER_WINDOW",
                        "trigger_window": trigger_label,
                        "segments_range": [i + 1, i + len(window_slice)],
                        "text_snippet": (w_clean[:140] + "...") if len(w_clean) > 140 else w_clean,
                        "full_text": w_clean,
                        "semantic_attack_score": w_semantic,
                        "layout_risk": w_max_struct,
                        "final_risk": w_final_risk,
                        "risk_score": w_final_risk,
                        "action": w_action,
                        "heuristic_category": w_distil_res.get("heuristic_category"),
                        "distilbert_class": w_distil_res.get("heuristic_category") or w_distil_res.get("verdict", "INJECTION"),
                        "anomaly_codes": ["SPLIT_PAYLOAD_DETECTED_IN_WINDOW"],
                        "reason": f"Adversarial instruction detected across reading-order window ({trigger_label})",
                        "is_malicious": True,
                    }
                    window_triggers.append(w_threat)
                    flagged_threats.append(w_threat)
                    if w_final_risk > max_risk:
                        max_risk = w_final_risk

        # Compute page risk heatmap
        pages_summary = []
        for p in sorted(page_risks.keys()):
            scores = page_risks[p]
            p_max = max(scores) if scores else 0
            p_avg = sum(scores) / len(scores) if scores else 0
            pages_summary.append({
                "page": p,
                "max_risk": p_max,
                "avg_risk": round(p_avg, 1),
                "threat_count": sum(1 for s in segments_results if s["page"] == p and s["is_malicious"]),
                "status": "DANGER" if p_max >= settings.block_risk_threshold else ("WARNING" if p_max >= settings.review_risk_threshold else "CLEAN")
            })

        if any(s["action"] == "BLOCK" for s in segments_results) or any(w["action"] == "BLOCK" for w in window_triggers):
            overall_action = "BLOCK"
        elif any(s["action"] == "REVIEW" for s in segments_results) or any(w["action"] == "REVIEW" for w in window_triggers):
            overall_action = "REVIEW"
        else:
            overall_action = "PASS"

        # Determine primary heuristic category from flagged threats
        primary_heuristic = "BENIGN"
        for s in flagged_threats:
            h = s.get("heuristic_category")
            if h and h != "UNKNOWN":
                primary_heuristic = h
                break
        if primary_heuristic == "BENIGN" and flagged_threats:
            primary_heuristic = "INJECTION"

        return {
            "filename": filename,
            "total_pages": len(page_risks),
            "total_segments": len(segments_results),
            "total_characters": extracted.raw_character_count,
            "cloaked_segments_count": sum(1 for s in segments_results if s["is_hidden"]),
            "flagged_threats_count": len(flagged_threats),
            "overall_risk_score": max_risk,
            "overall_action": overall_action,
            "overall_status": "MALICIOUS (BLOCKED)" if overall_action == "BLOCK" else ("SUSPICIOUS (REVIEW)" if overall_action == "REVIEW" else "CLEAN & SAFE"),
            "primary_heuristic_category": primary_heuristic,
            "pages_summary": pages_summary,
            "flagged_threats": flagged_threats,
            "window_triggers": window_triggers,
            "all_segments": segments_results,
            "sanitized_document_text": "\n\n".join(clean_text_segments),
            "warnings": extracted.extraction_warnings
        }

document_scanner = DocumentScanner()
