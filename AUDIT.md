# Phase 0 Architecture Audit: PromptShield X

**Date:** 2026-10-06  
**Scope:** Detection Pipeline, Classification Logic, Structural Extraction, Risk Synthesis, Isolation Forest, and Normalization.  
**Constraint Enforced:** Read-only audit; zero code modifications performed in this phase.

---

## 1. Label Assignment: Origin & Mapping Mechanics

### Question
> *Where is the final label (e.g. `JAILBREAK_PROMPT_LEAKAGE`) assigned? Does the binary model emit it, or do regex rules/keywords map it?*

### Findings
1. **The DistilBERT Model is Strictly Binary:**
   - The model checkpoint deployed at `app/modules/weights/distilbert/model.safetensors` contains a classification head with:
     - `classifier.weight`: shape `torch.Size([2, 768])`
     - `classifier.bias`: shape `torch.Size([2])`
   - The underlying PyTorch model outputs exactly **2 raw logits**, corresponding to:
     - Class `0`: Benign / Safe
     - Class `1`: Prompt Injection
   - **The binary neural model never emits `JAILBREAK_PROMPT_LEAKAGE` or `INDIRECT`.**

2. **Where and How the Specific Attack Labels are Assigned:**
   - **Path A: `app/modules/distilbert_classifier.py` (Lines 117–134 & 169–176):**
     - When `len(probs) == 2`, the model's raw binary index `1` is initially named `"DIRECT"`.
     - Then, a regex/keyword override intercepts the prediction at line 169:
       ```python
       elif has_directive:
           if category == "BENIGN" or confidence < 0.85:
               tl = text.lower()
               category = "JAILBREAK_PROMPT_LEAKAGE" if ("dan" in tl or "mode" in tl or "bypass" in tl) else "DIRECT"
               confidence = 0.95
               raw_scores = {category: 0.95, "BENIGN": 0.03, "INDIRECT": 0.01, "DIRECT": 0.01}
       ```
     - Substring checks on `"dan"`, `"mode"`, and `"bypass"` fabricate the label `"JAILBREAK_PROMPT_LEAKAGE"` and synthesize synthetic `raw_scores` (0.95, 0.03, 0.01, 0.01).
   - **Path B: `app/modules/semantic_classifier.py` (Lines 120–153 - Fallback):**
     - When local DistilBERT is bypassed or unavailable, `classify_prompt_local_fallback()` performs keyword checks:
       - `"system prompt"` / `"reveal your prompt"` $\to$ `"prompt_extraction"`
       - `"dan"` / `"bypass safety"` / `"no restrictions"` $\to$ `"jailbreak"`
       - `"agent"` / `"override"` $\to$ `"agent_manipulation"`
       - `"ignore previous"` $\to$ `"prompt_injection"`
   - **Path C: `app/api/routes.py` (Lines 145–148):**
     - For document scans:
       ```python
       attack_category = "INDIRECT_PDF_INJECTION" if result["flagged_threats_count"] > 0 else "SAFE_DOCUMENT"
       ```
       This assigns the label entirely based on whether `flagged_threats_count > 0`.

**Conclusion:** The neural model is purely binary. All fine-grained categorization tags (`JAILBREAK_PROMPT_LEAKAGE`, `INDIRECT`, `prompt_extraction`, `agent_manipulation`) are produced by heuristic regexes, string matching, and synthetic score dictionary construction.

---

## 2. Model Override Analysis in `distilbert_classifier.py`

### Question
> *In `distilbert_classifier.py`, can `DIRECTIVE_PATTERNS` / `META_DISCUSSION_REGEX` override or replace the model's output? List every override path.*

### Findings
Yes. The model's raw probability distribution is intercepted and replaced by three distinct conditional override branches in lines 162–183:

```
[Raw Model Output: logits -> softmax -> (p_benign, p_inj)]
                         │
                         ▼
        ┌────────────────────────────────┐
        │ Has Meta-Discussion Match?    │
        │ (META_DISCUSSION_REGEX)        │
        └────────────────┬───────────────┘
                         │
         YES & no payload│                     NO
         ────────────────┼──────────────────────┐
         │                                      │
         ▼                                      ▼
[OVERRIDE PATH 1]                    ┌────────────────────┐
Force Category = "BENIGN"            │ Has Directive Match?│
Force Confidence = 0.95              │ (DIRECTIVE_PATTERNS)│
Fabricate raw_scores:                └─────────┬──────────┘
  BENIGN: 0.95                                 │
  DIRECT: 0.02                    YES          │          NO
  INDIRECT: 0.02            ┌──────────────────┴──────────────────┐
  JAILBREAK: 0.01           ▼                                     ▼
                   [OVERRIDE PATH 2]                     [OVERRIDE PATH 3]
                   If model said BENIGN                  If model said ATTACK
                   or confidence < 0.85:                 (category != "BENIGN"):
                   Force Category = JAILBREAK / DIRECT   Force Category = "BENIGN"
                   Force Confidence = 0.95               Force Confidence = 0.95
                   Fabricate raw_scores:                 Fabricate raw_scores:
                     Attack: 0.95                          BENIGN: 0.95
                     BENIGN: 0.03                          Others: ~0.01 - 0.02
```

### Complete List of Override Paths:

1. **Override Path 1: Meta-Discussion Downgrade (`lines 163–167`):**
   - **Condition:** `has_meta and not has_payload`
   - **Effect:** Even if the neural model predicts injection with 99.9% confidence, the prediction is completely discarded. The output is forced to `category = "BENIGN"`, `confidence = 0.95`, and synthetic `raw_scores` are returned.
2. **Override Path 2: Adversarial Directive Elevation (`lines 169–176`):**
   - **Condition:** `has_directive and (category == "BENIGN" or confidence < 0.85)`
   - **Effect:** Even if the neural model predicts benign with 99.9% confidence, the model prediction is discarded. The category is forced to `"JAILBREAK_PROMPT_LEAKAGE"` or `"DIRECT"`, confidence is forced to `0.95`, and synthetic scores are returned.
3. **Override Path 3: Absence of Directive Downgrade (`lines 177–183`):**
   - **Condition:** `category != "BENIGN" and not has_directive`
   - **Effect:** If the neural model detects an attack but the text does not trigger any of the hardcoded `DIRECTIVE_PATTERNS` regexes, the model's detection is deemed a false positive and forced back to `category = "BENIGN"` with `confidence = 0.95`.

**Implication:** In current execution, the neural model's prediction only survives if it already agrees with the heuristic regexes. The heuristic rules act as an authoritative supervisor over the transformer.

---

## 3. Structural vs. Semantic Score Combination & Gating

### Question
> *Exactly how structural scores (font size, color, off-canvas, metadata) combine with the semantic score. Is it additive? Can structure alone reach BLOCK?*

### Findings
The scoring mechanism is implemented in `app/modules/document_scanner.py` (lines 58–133):

#### 1. Component Score Derivations
* **Semantic Attack Score ($S_{\text{semantic}}$):** 
  - If `category == "BENIGN"`: $S_{\text{semantic}} = \max(\text{round}((1 - \text{conf}) \times 15), \text{pattern\_severity})$.
  - If `category != "BENIGN"`: $S_{\text{semantic}} = \max(\text{pattern\_severity}, \text{round}(\text{base\_sev} \times \text{conf}))$, where $\text{base\_sev} \in \{80, 90\}$.
* **Structural Anomaly Scores:**
  - $S_{\text{layout}}$: font size $<2.0\text{pt}$ yields $30.0$ ($45.0$ if $\le 1.5\text{pt}$).
  - $S_{\text{visibility}}$: white text (color `0xFFFFFF`) yields $50.0$.
  - $S_{\text{position}}$: off-canvas coordinates outside page bounds yield $40.0$.
  - $S_{\text{metadata}}$: metadata payload regex match yields $50.0$.

#### 2. Synthesis Formulation
In `lines 105–111`:
$$\text{RawCombined} = \text{round}\Big(0.60 \times S_{\text{semantic}} + 0.20 \times S_{\text{layout}} + 0.10 \times S_{\text{visibility}} + 0.05 \times S_{\text{position}} + 0.05 \times S_{\text{metadata}}\Big)$$
* This is a **weighted linear additive combination**, with weights summing to $1.00$.

#### 3. Semantic Gating Rules (`lines 116–133`)
```python
if semantic_attack_score <= 25:
    final_risk = min(25, raw_combined)
    action = "PASS"
elif semantic_attack_score <= 60:
    final_risk = min(60, raw_combined)
    action = "REWRITE" if final_risk >= 30 else "PASS"
else:
    if seg.is_hidden or seg.cloaking_signal != "NONE":
        final_risk = max(88, min(100, raw_combined + 30))
    else:
        final_risk = max(70, min(100, raw_combined))
    action = "BLOCK" if final_risk >= 60 else "REWRITE"
```

#### Can Structure Alone Reach BLOCK?
* **Under the current gated implementation:** **No.** If $S_{\text{semantic}} \le 25$, the gating branch unconditionally enforces:
  $$\text{final\_risk} = \min(25, \text{raw\_combined}) \le 25 \implies \text{action} = \text{"PASS"}$$
* **However, the vulnerability exists at the gate threshold ($S_{\text{semantic}} > 25$):**
  - If out-of-domain benign text (or an uncalibrated model output) yields $S_{\text{semantic}} = 26$ to $60$:
  - Extreme layout anomalies can boost $\text{RawCombined}$ up to $60$, triggering `REWRITE`.
  - In `lines 127–129`, when $S_{\text{semantic}} > 60$, an additive penalty of **$+30$** is added directly to `raw_combined` (`raw_combined + 30`) with an artificial floor of `88`, forcing `BLOCK`.

---

## 4. Measured Execution Trace on Benign Document & Current 74/100 Reproduction Status

### Question
> *Provide a real measured trace on the status-report PDF on current code with actual numbers per segment. State clearly whether the 74/100 BLOCK still reproduces on current code.*

### Experimental Execution Setup
A test document was synthesized with the exact user text and realistic PDF metadata:
- **Document Text:**
  ```text
  Project Status Report
  This is a normal business document for security testing.
  The engineering team reviewed access controls and logging.
  No unusual incidents were reported during the review period.
  Next steps include documentation updates and routine maintenance.
  ```
- **Document Metadata:** `/Producer`: `"PyMuPDF 1.24.0"`, `/Author`: `"Nilay"`, `/CreationDate`: `"D:20260925162000"`, `/Title`: `"Project Status Report"`.

### Measured Execution Log (Current Codebase):

```text
=== MEASURED TRACE ON CURRENT CODE ===
Total segments extracted: 1

--- SEGMENT 1 ---
Segment Type: DOCUMENT_TEXT | Cloaking Signal: NONE | is_hidden: False
Text: Project Status Report This is a normal business document for security testing. The engineering team reviewed access controls and logging. No unusual incidents were reported during the review period. Next steps include documentation updates and routine maintenance.
RAW Model Probabilities: p_benign=0.1449, p_inj=0.8551
Model Argmax: INJECTION (conf=0.8551)
Fired Directive Rules: []
Fired Meta Regex: False, Fired Payload Regex: False
Pattern Scanner Matches: [] (severity=0)
Structural Sub-scores: layout=0.0, visibility=0.0, position=0.0, metadata=0.0
Computed Semantic Attack Score: 1
Raw Combined Score: 1
Gate Tier Applied: Tier 1 (sem <= 25 -> capped at 25, PASS)
Penalties Applied: confidence_penalty=0.0, hidden_bonus=0
Final Risk: 1
Action: PASS
Reason: Standard benign document content

=== OVERALL DOCUMENT VERDICT ===
Overall Risk: 1 / 100
Overall Action: PASS
Overall Status: CLEAN & SAFE
Cloaked Segments Count: 0
Flagged Threats Count: 0
```

### Critical Empirical Findings:

1. **Does the 74/100 BLOCK still reproduce on the CURRENT code?**
   - **NO.** The 74/100 BLOCK does **not** reproduce on the current codebase. The document scores **1 / 100 PASS**.
2. **HOWEVER, Why Did It Pass? The Underlying Model Actually Failed:**
   - Notice the raw neural model probability:
     $$\mathbf{p_{\text{benign}} = 0.1449}, \quad \mathbf{p_{\text{inj}} = 0.8551}$$
   - **The raw fine-tuned DistilBERT model alone falsely classified this standard business paragraph as an INJECTION with 85.51% confidence!**
   - It only passed because **Override Path 3** in `distilbert_classifier.py` intercepted the model's false-positive:
     ```python
     elif category != "BENIGN" and not has_directive:
         category = "BENIGN"
         confidence = 0.95
         raw_scores = {"BENIGN": 0.95, "DIRECT": 0.02, ...}
     ```
     Because no string in `DIRECTIVE_PATTERNS` fired, the code discarded the model's `0.8551` injection prediction and forced `category = "BENIGN"`, computing $S_{\text{semantic}} = \text{round}((1 - 0.95) \times 15) = 1$.
   - **Conclusion:** The current code passes this benign test **solely because of heuristic regex suppression masking the model's false-positive**. If model overrides are turned off in Phase 1, this paragraph will immediately be flagged by the raw model unless properly calibrated and normalized.

---

## 5. Catalog of Hardcoded Typo & Leetspeak Maps

### Comprehensive Inventory Across the Repository:

1. **`app/modules/distilbert_classifier.py` (`lines 78–91`):**
   - **Leetspeak Transliteration Table:**
     ```python
     trans_map = str.maketrans({
         '3': 'e', '0': 'o', '1': 'i', '4': 'a', '@': 'a',
         '$': 's', '5': 's', '!': 'i', '7': 't', '+': 't', '8': 'b'
     })
     ```
   - **Explicit Typoglycemia / Typo-Squatting Replacements:**
     ```python
     deob = re.sub(r'\bignroe\b', 'ignore', deob, flags=re.I)
     deob = re.sub(r'\brevael\b', 'reveal', deob, flags=re.I)
     deob = re.sub(r'\bpr0mpt\b', 'prompt', deob, flags=re.I)
     deob = re.sub(r'\bsyst3m\b', 'system', deob, flags=re.I)
     deob = re.sub(r'\bdir3ctives?\b', 'directive', deob, flags=re.I)
     ```
   - **Zero-Width / Invisible Character Stripping:**
     ```python
     cleaned = re.sub(r'[\u200B-\u200D\uFEFF\u2060\u00A0]', ' ', text)
     ```

2. **`app/modules/sanitizer.py` (`lines 62–72`):**
   - **Hardcoded Invisible Unicode Codepoints:**
     ```python
     invisible_chars = [
         "\u200b",  # zero-width space
         "\u200c",  # zero-width non-joiner
         "\u200d",  # zero-width joiner
         "\u2060",  # word joiner
         "\ufeff",  # BOM / zero-width no-break space
     ]
     ```

3. **`app/modules/extractors/pdf_extractor.py` (`lines 61–63`):**
   - **Hardcoded Suspicious Metadata Substring Regex:**
     ```python
     SUSPICIOUS_METADATA_REGEX = re.compile(
         r"(?i)\b(ignore|disregard|system\s*prompt|you are now|dan|override|bypass|reveal)\b"
     )
     ```

4. **`app/modules/semantic_classifier.py` (`lines 124–137`):**
   - **Hardcoded Fallback Trigger Phrases:**
     ```python
     "system prompt" in tl or "reveal your prompt" in tl or "reveal prompt" in tl
     "ignore previous" in tl or "ignore all previous" in tl or "ignore instructions" in tl
     "dan" in tl or "bypass safety" in tl or "no restrictions" in tl or "act as" in tl
     "agent" in tl or "override" in tl
     ```

---

## 6. Isolation Forest Audit (`anomaly_detector.py`)

### Question
> *Document how `anomaly_detector.py` contributes to the composite score in the live Prompt/RAG firewall: inputs, weight, whether it can alone change the action.*

### Findings:

1. **Architecture & Mechanism:**
   - **Embedding Input:** Dense text representations generated by `SentenceTransformer("all-MiniLM-L6-v2")` (384-dimensional vector).
   - **Model:** `sklearn.ensemble.IsolationForest` serialized at `app/modules/weights/isolation_forest.pkl`.
   - **Score Output:** Decision function is scaled to $[0, 100]$:
     $$\text{norm\_score} = \max\Big(0, \min\big(100, \text{int}((0.3 - \text{score\_raw}) \times 100)\big)\Big)$$
     `is_anomalous` is flagged if `pred == -1` or `norm_score > 60`.

2. **Contribution to Live Firewall Endpoints:**
   - **Endpoint 1: `/analyze` (Live User Prompt Firewall):**
     - In `app/api/routes.py` (`lines 39–74`):
       ```python
       pattern_result = scan_prompt(clean_prompt)
       classification = classify_prompt(clean_prompt)
       scored = compute_risk_score(pattern_result["severity"], classification)
       ```
       `detect_anomaly` is **never imported and never called**. **Weight = 0.00%.**
   - **Endpoint 2: `/analyze-rag` (Live RAG Chunk Firewall):**
     - In `app/api/routes.py` (`lines 76–119`):
       `detect_anomaly` is **never imported and never called**. Line 81 explicitly acknowledges: *"Full version (6.4-6.8: chunk scanner, anomaly detector, reliability filter, evaluator LLM) is documented as future work."* **Weight = 0.00%.**
   - **Endpoint 3: `/analyze-pdf` (Live Document Scanner):**
     - In `app/modules/document_scanner.py`: `from app.modules.anomaly_detector import detect_anomaly` exists at line 21, but `detect_anomaly` is **never invoked** anywhere inside `scan_pdf_bytes`. **Weight = 0.00%.**
   - **Risk Engine:**
     - In `app/core/risk_engine.py`: `compute_risk_score(pattern_severity, classification)` only accepts pattern severity and classification confidence. It does not accept or compute anomaly scores.
   - **Streamlit UI:**
     - In `streamlit_app.py` (`line 343`), `anomaly = detect_anomaly(clean)` is called solely to render an anomaly score metric badge on the frontend Playground tab.

3. **Can Isolation Forest Alone Change the Action?**
   - **ABSOLUTELY NOT.** The Isolation Forest model has **zero impact** on risk scores and can **never** trigger `PASS`, `REWRITE`, or `BLOCK` in any live API endpoint. It is currently dormant backend scaffolding.

---

## Comprehensive Audit Summary

| Audit Dimension | Current Code State | Academic & Architectural Finding |
| :--- | :--- | :--- |
| **Model Classification** | Purely binary ($[2, 768]$). | Category tags are synthetic heuristic constructs. |
| **Model Overrides** | 3 conditional branches override model. | Model output is overridden whenever it disagrees with regexes. |
| **Score Synthesis** | Weighted linear sum ($0.60 S_{\text{sem}} + \dots$). | Hand-tuned weights; gating prevents layout-only block. |
| **Benign Status PDF** | Scores **1 / 100 PASS**. | Raw model flagged injection ($p=0.8551$); saved by Override Path 3. |
| **74/100 Reproduction** | Does **NOT** reproduce on current code. | Previous FP was caused by raw metadata parsed as hidden + penalties. |
| **Isolation Forest** | Dormant (Weight = 0.00%). | Completely uncoupled from all live risk equations. |
| **Obfuscation Filter** | Static leetspeak map + 5 word regexes. | Narrow pattern overfitting; fails on edit-distance variants. |

---

*AUDIT.md update complete. Awaiting confirmation to execute Phase 1.*
