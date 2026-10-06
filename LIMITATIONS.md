# PromptShield X — Architectural & Empirical Limitations

This document honestly details the measured empirical failure modes, statistical vulnerabilities, and operational boundaries of PromptShield X discovered during rigorous Phase 4 auditing.

---

## 1. Static Regex / Keyword Rules
- **Rule Efficacy**: Standalone regex rules have near-zero utility against modern prompt injections:
  - **AUROC**: **0.5495** (barely above random guessing at 0.50).
  - **Recall at $\le 1\%$ FPR**: **0.00%**.
- **False-Positive Amplification**: 
  - On the held-out test set, the neural model alone achieves **0.00% FPR** on enterprise hard negatives (legal text, footnotes, disclaimers).
  - Blending static rules in default `combined` mode increases hard-negative FPR to **9.33%** (14 false positives), purely due to literal substrings matching rules like `role-override` or `end-of-prompt-marker`.
  - When `rules_evidence_only=True` is enabled, rule hits can raise severity to `REVIEW` at most ($\le 40$), restoring hard-negative FPR to **0.00%**.
- **Conclusion**: Hardcoded regex pattern matching does not enhance neural detection recall on novel injections and functions primarily as a liability that drives false alarms on complex enterprise prose.

---

## 2. Context Dilution Vulnerability (Long Inputs)
- **Token Length Dilution**: When adversarial payloads are placed at the end of long benign texts, the raw attention mechanism of DistilBERT rapidly dilutes the attack signal across tokens:
  - 12 tokens (Attack only): $p_{\text{inj}} = 0.9970$ (BLOCK)
  - 82 tokens (10 benign sentences + attack): $p_{\text{inj}} = 0.8550$ (BLOCK)
  - 152 tokens (20 benign sentences + attack): $p_{\text{inj}} = 0.1147$ (DROPS TO PASS)
  - 512 tokens (80 benign sentences + attack): $p_{\text{inj}} = 0.0209$ (SAFE PASS)
- **Mitigation & Remaining Gap**: Overlapping sliding-window chunking (stride 256, max length 512) evaluates each slice independently. However, if an attack directive itself is blended into 300+ tokens of surrounding prose within a single window, neural confidence drops significantly unless structural cloaking anomalies or rule hits compensate.

---

## 3. Meta-Discussion & Quoting
- **Isolated Meta-Discussion Sentences**: Benign sentences discussing attacks in academic, compliance, or security training contexts (e.g. *"The training curriculum discusses why attackers use 'ignore previous instructions'"*) yield a **48.0% non-PASS rate** (22% BLOCK, 26% REVIEW) in isolation because the model reacts to the high density of adversarial vocabulary.
- **Prose Context Resolution**: When embedded in full paragraph context (surrounding explanatory prose), BLOCK drops to **0.0%**, with 38% REVIEW and 62% PASS.
- **Quote Evasion Safeguard**: Wrapping real injection attacks in quotation marks (`"..."`), markdown code blocks (```` ``` ````), or prefixes like `"For example:"` does **not** lower their semantic attack score (raw attacks score $p_{\text{inj}} \approx 0.99$, and quoted variants score identically $p_{\text{inj}} \approx 0.99$). Quoted context softening only reduces rule-derived penalties, never neural probability.

---

## 4. Off-the-Shelf Baseline Generalization Deficits
- **Deepset DeBERTa (`deepset/deberta-v3-base-injection`)**: Suffers from extreme false-positive collapse on enterprise English:
  - Default FPR on diverse user queries and corporate prose is **62.65%**, with an Expected Calibration Error of **0.4737**.
  - While trained on synthetic adversarial datasets, it flags legitimate financial footnotes ($p_{\text{inj}} = 0.9969$) and confidentiality notices ($p_{\text{inj}} = 0.9399$) as malicious injections.
- **ProtectAI DeBERTa (`protectai/deberta-v3-base-prompt-injection-v2`)**: Demonstrates higher precision (3.36% default FPR), but achieves only **57.08% recall at $\le 1\%$ FPR** and **65.57% at $\le 5\%$ FPR** on unseen indirect and agent-oriented prompt injections.
