# PromptShield X: Evaluation & Retraining Datasets Specification

**Date:** 2026-10-06  
**Scope:** Phase 3 Dataset Provenance, Hard-Negative Engineering, Source-Based Splitting, and Licensing.

---

## 1. Public Prompt Injection Datasets & Licenses

| Dataset Identifier | Primary Modality / Attack Families | License | Sample Count | Role in Phase 3 |
| :--- | :--- | :--- | :---: | :--- |
| **`deepset/prompt-injections`** | Direct injections, extraction, task overrides | **Apache 2.0** | 662 (546 train, 116 val/test) | Core train & val corpus |
| **`rubend18/ChatGPT-Jailbreak-Prompts`** | Roleplay bypasses, persona hijacking, DAN variants | **MIT** | 79 | Train corpus (jailbreak diversity) |
| **`JasperLS/prompt-injections`** | Public benchmark prompt injection variations | **Apache 2.0** | 662 | Source cross-validation |
| **`InjecAgent & HackAPrompt / Synthetic Held-out`** | Indirect injection, tool manipulation, extraction | **MIT / Apache 2.0** | 150 | **Held-out Test Source (Touched ONCE)** |

> [!IMPORTANT]
> **Excluded Datasets:**
> `AdvBench` and `BeaverTails` are explicitly **excluded**. They represent general harmful content / toxicity / safety jailbreaks (e.g. bomb-making, hate speech), which are the domain of downstream LLM safety alignments, not prompt-injection or document-layer firewalls.

---

## 2. Benign Hard-Negative Corpus

Standard neural models overfit heavily to imperative keywords (*"must"*, *"instructions"*, *"ignore"*, *"policy"*, *"report"*). To resolve the 90% false positive rate identified in Phase 1.5, we constructed a dedicated **Hard-Negative Corpus** (800+ total samples across train, val, and test):

1. **Footnotes & Disclaimers:**
   - Financial statement footnotes (*"Footnote 1: Amounts are presented in thousands unless otherwise stated"*).
   - Legal disclaimers, liability limitations, copyright, and confidentiality notices.
2. **Terms of Service & Privacy Agreements:**
   - User compliance requirements, data retention policies, and account termination clauses.
3. **Imperative Business & Operations Text:**
   - Directives directed at employees (*"You must submit expense reports by Friday"*, *"Do not distribute keys to external parties"*).
4. **Technical Documentation & Runbooks:**
   - Migration scripts, daemon configuration, API contracts, SQL execution directives.
5. **Meta-Discussion & Security Training (Labeled Benign):**
   - Articles and curriculum discussing injection (*"During employee training, instructors discuss why attackers attempt to ignore previous instructions"*).

---

## 3. Source-Based Partitioning (Train / Val / Test)

Partitions are split strictly by **SOURCE ORIGIN** so that test sources are completely unseen during training:

```
[Total Dataset Pool: ~2,500 samples]
  │
  ├── TRAIN SET (~1,550 samples)
  │    ├── Deepset Train (270 Injection, 276 Benign)
  │    ├── Rubend18 Jailbreaks (79 Injection)
  │    ├── Augmented Perturbations (Train attacks augmented with homoglyph, leet, zero-width)
  │    └── Training Hard Negatives (700 Benign: enterprise policies, technical docs, imperative text)
  │
  ├── VALIDATION SET (~300 samples)
  │    ├── Deepset Validation subset (60 samples)
  │    ├── Validation Jailbreak templates (40 Injection)
  │    └── Validation Compliance & Meta-discussion (200 Benign)
  │    └── Used strictly for model checkpointing & threshold tuning (targeting FPR <= 1.0%)
  │
  └── TEST SET (400 samples) - TOUCHED ONCE
       ├── Held-out Novel Injection Sources (150 Injection)
       ├── Held-out Enterprise Hard Negatives (150 Benign: footnotes, legal, technical)
       └── Held-out Standard Benign Queries (100 Benign: general assistance)
```

---

## 4. Training Augmentation Constraints

- **Perturbation augmentation (homoglyph, leetspeak, typoglycemia, zero-width) is applied strictly to the TRAIN partition.**
- The validation and held-out test partitions remain unaugmented in their source distributions to measure true generalization.

---

## 5. v36: real-world documents (replaces v35 for all new results)

Goal: data and evaluation that can support the claim **"the detector works on real-world documents."**

```
python eval/build_dataset_v36.py          # deterministic (seed 42); downloads raw sources to eval/data/raw/
# train on Colab: notebooks/colab_train_v36.ipynb  (or: python eval/train_v36.py --seeds 42 123 456)
python eval/evaluate_v36.py               # text-level dev test (heldout_test_v36)
python eval/evaluate_documents_v36.py     # rendered PDF / HTML documents through the app's extractors
python eval/evaluate_v36.py --final       # ONCE, at the very end
python eval/evaluate_documents_v36.py --final   # ONCE, at the very end
```

Counts, removals and per-source label balance: `eval/data/dataset_report_v36.json`.

### 5.1 What the v35 audit found and what v36 does

| v35 problem | v36 fix |
| :--- | :--- |
| Source almost determined the label | Every document genre appears with both labels: ~30% of real document chunks get an attack from the same split's attack pool at a sentence boundary; BIPIA documents appear clean and attacked |
| Style/length LR reached 0.899 AUROC | Style-only LR is 0.54 on heldout_test_v36 (chance) |
| Template hard negatives (157 unique strings) | Replaced by ~7,000 real document chunks; templates kept only as a probe (`diagnostic_templates_v36.json`) |
| "act as" label conflicts | Standalone persona prompts labeled 1 moved to `ambiguous_v36.json`; persona after an unrelated question stays 1; benign persona rows flagged `persona=True` |
| Standalone BIPIA strings are injections only in context | Used only inside their real host documents (`context_dependent=True`) |
| 13-18% duplicate rows | Exact + 8-gram dedupe within and across splits; label conflicts dropped; CIs bootstrap over `group` |
| No long / padded controls | Long benign docs, padded benign instructions and buried attacks in **val and test**; long-document thresholds are chosen on val |
| No real indirect attacks; test reused | `final_test_v36` built from sources never used elsewhere (LLMail-Inject, InjecAgent, BBC, jackhhao, no_robots), lock-file enforced |
| Gandalf dominated attacks | Capped at 300 in train; Tensor Trust hijacking/extraction attacks added |

### 5.2 Sources

| Role | Source | License | Splits |
| :--- | :--- | :--- | :--- |
| Attacks | deepset/prompt-injections | Apache-2.0 | train 85% / val 15% (by text) / test = upstream test |
| Attacks | Lakera/gandalf_ignore_instructions | MIT | upstream splits, train capped at 300 |
| Attacks | rubend18/ChatGPT-Jailbreak-Prompts | MIT | train only |
| Attacks | qxcv/tensor-trust (hijacking + extraction benchmarks) | see dataset card | 70/15/15 by unique attack |
| Benign prompts | fka/awesome-chatgpt-prompts, databricks-dolly-15k | CC0 / CC-BY-SA | 65/15/20, disjoint |
| Real documents | LEDGAR + UNFAIR-ToS (coastalcph/lex_glue) | CC-BY-4.0 / see card | upstream splits |
| Real documents | Enron emails (Yale-LILY/aeslc) | see card | upstream splits |
| Real documents | Wikipedia (Salesforce/wikitext, wikitext-2-raw) | CC-BY-SA | upstream splits |
| Real documents | arXiv ML abstracts (CShorten/ML-ArXiv-Papers) | AFL-3.0 | 1,200 / 300 / 400 |
| **Meta-discussion** | ~2,000 real arXiv abstracts about prompt injection / jailbreaks (arXiv API, cached in `raw/arxiv_meta.json`) | arXiv metadata (CC0) | 60/15/25 by paper |
| Context-dependent | geodesic-research/bipia host emails / tables / code | see card | 70/15/15 by document, attacks disjoint |
| **Final only** | microsoft/llmail-inject-challenge phase 2: real attacker emails (`api_triggered` = fired against a real LLM, and judge-labeled) + challenge benign emails | MIT | final |
| **Final only** | InjecAgent tool responses (DH + DS base) | MIT | final |
| **Final only** | SetFit/bbc-news (unseen genre, 30% with an LLMail attack embedded), jackhhao/jailbreak-classification, HuggingFaceH4/no_robots | see cards | final |

HackAPrompt is gated on the Hub and was not used. Request access and add it to the attack pools if wanted.

### 5.3 Files

| File | Role |
| :--- | :--- |
| `train_v36.json` | training (+20% perturbation augmentation, which keeps the original's `group`) |
| `val_v36.json` | epoch selection and **all** thresholds |
| `heldout_test_v36.json` | dev test, reported every run |
| `final_test_v36.json` | **touch once** |
| `format_suite_v36_val.json` | rendered documents for choosing the document-level threshold |
| `format_suite_v36_heldout.json` | rendered documents, dev |
| `format_suite_v36_final.json` | rendered documents from BBC + LLMail emails with LLMail attacks, **touch once** |
| `ambiguous_v36.json`, `diagnostic_templates_v36.json` | probes, never trained on |

Row fields: `text, label, source, category, group, persona, context_dependent, doc_injected, is_long, augmented, meta_discussion`.

**Rendered-document suites.** Each document is rendered as a real PDF (PyMuPDF) and HTML page five ways:
- PDF: visible, micro-font, white text, off-canvas, metadata
- HTML: visible, `display:none`, comment, `aria-hidden`, zero-width

Each rendering appears once with an attack and once with **benign text hidden the same way** (a real sentence from another document), so hiding alone cannot separate the classes. The app's own `PDFExtractor` / `WebExtractor` extract the text, and the document score is the max over segments. The production `DocumentScanner` is also reported for PDFs.

### 5.4 Labeling convention

Label 1 means the text, placed in a model's context (user turn or retrieved document), tries to override or redirect the system's instructions. A user asking for a persona is 0. A persona used to switch tasks mid-input, or combined with removing restrictions, is 1. Text that *discusses* injection (papers, training material) is 0.

### 5.5 The claim protocol

The claim "works on real-world documents" may be made only from one run of `--final` (both scripts), with every threshold chosen on val, and must quote:
1. Recall on `llmail_attack_api_triggered`, `llmail_context_attacked`, `injecagent_*`, and rendered `format_suite_v36_final` attacks (per hiding method).
2. FPR on `llmail_benign_email`, `llmail_context_benign`, BBC benign, `no_robots`, and rendered benign documents.
3. 95% bootstrap CIs over `group`, and the TF-IDF baseline next to every number.

### 5.6 Baseline measurements on heldout (2026-10-06, before retraining)

| Scorer | AUROC all | AUROC source-balanced | Recall, attacks in documents @val FPR 5% | Enron FPR | Visible PDF attack recall (rendered) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Style-only LR | 0.54 | 0.54 | 0.01 | 0.01 | - |
| TF-IDF LR (trained on v36) | 0.96 | 0.94 | 0.84 | 0.08 | 0.75 |
| Production DistilBERT (trained on v35) | 0.80 | 0.72 | 0.42 | 0.41 | 0.55 |

The v35 model does **not** yet work on real documents. Retrain on v36 before making any claim.

**After retraining on v36 (Colab, best of 3 seeds), dev only:** AUROC 0.993 (source-balanced 0.990), attacks in documents 96% caught at the val 1% threshold, rendered PDF+HTML documents 91% caught at 1.9% FPR. The production pipeline with v36 weights caught 96.9% at 1.0% FPR on val PDFs (default thresholds kept) and 95.8% at 2.2% FPR on heldout PDFs.

### 5.8 v36 FINAL TEST RESULT (run once, 2026-10-07): the claim FAILS

| final_test_v36 (unseen sources) | TF-IDF | v35 | v36 |
| :--- | :---: | :---: | :---: |
| AUROC, all | 0.50 | 0.49 | 0.45 |
| LLMail api_triggered recall @val 1% / 5% | 3% / 8% | 8% / 22% | 4% / 27% |
| LLMail 10-email contexts recall @val 1% / 5% | 2% / 7% | 16% / 23% | 5% / 5% |
| InjecAgent recall | ~1% | ~1% | 0-4% |
| BBC benign FPR @val 1% / 5% | 0.5% / 6% | 2% / 13% | 8.5% / 96% |
| Rendered final docs (format_suite_v36_final) recall / FPR @val 5% | 22% / 14% | 24% / 13% | 54% / 40% |

The dev results were real but optimistic: train, val and test only contained "ignore your instructions"-style attack families, while real indirect attacks are persuasive emails and polite tool requests. The v36 final is spent. Files: `eval/results/v36_eval_final.json`, `format_suite_v36_final_results.json`.

### 5.7 Known limitations

- InjecAgent has no benign twin, so it gives recall only.
- Attacks embedded in documents in train/val/test reuse that split's attack strings. Novel attacker phrasing is measured only by LLMail in the final test.
- BIPIA has ~150 host documents, so its CIs are wide.
- Rendered documents are generated, not scraped. Real-world PDF/HTML layouts (multi-column, scanned, tables) are not covered.
- English only, apart from some German in deepset.


---

## 6. v37: real indirect attacks (after the failed v36 final)

```
python eval/build_dataset_v37.py                     # builds on the v36 splits
python eval/train_v36.py --data v37 --seeds 42 123 456   # GPU (fp16, label smoothing 0.1) -> models/distilbert_v37
python eval/evaluate_v36.py --data v37
python eval/evaluate_documents_v36.py --data v37
python eval/evaluate_v36.py --data v37 --final       # ONCE
python eval/evaluate_documents_v36.py --data v37 --final   # ONCE
```

| Added to | Source | Why |
| :--- | :--- | :--- |
| train / val | LLMail-Inject **phase 1** real attacker emails (6,000 / 600 before dedupe) | real persuasive, adaptive indirect attacks |
| train / val / test | InjecAgent base, split by attacker instruction, each with a **benign twin** (same tool-response JSON, attacker slot filled with a real Enron sentence) | agent tool-output attacks; JSON format carries no signal |
| train / val / test | jackhhao (60/20/20), no_robots, BBC News (upstream splits, 30% with an LLMail attack embedded), LLMail benign emails | benign polite requests and news, so "please do X" is not an attack by itself |
| dev test | LLMail **phase 2** + its 10-email retrieval contexts | phase-1 to phase-2 generalization against stronger, blocklist-adaptive attackers |
| **final (touch once)** | **AgentDojo** (MIT): workspace / Slack / banking / travel environment text, benign defaults vs the same text with an injection vector filled by AgentDojo attacks (important_instructions, ignore_previous, system_message, injecagent, direct) | realistic agent tool outputs with benign twins; never used anywhere else |
| **final (touch once)** | PubMed abstracts (30% with an AgentDojo attack) and rendered PDF/HTML (`format_suite_v37_final`) | unseen benign genre |

Training changes: label smoothing 0.1 (v36 probabilities saturated, so val thresholds landed at 0.006 and were fragile), fp16 on GPU.

### 6.1 v37 dev results (heldout_test_v37, val-chosen thresholds)

AUROC 0.992 (source-balanced 0.991). At the val 1% threshold: LLMail phase-2 attacks 98.6% caught (v36: 0.4%), LLMail 10-email contexts 83%, InjecAgent unseen instructions 83%, benign LLMail emails/contexts 0% flagged; overall 91% recall at 0.9% FPR. Caveat: every LLMail attack targets contact@contact.com, so TF-IDF also reaches 94% there. Production pipeline with v37 on val PDFs (default thresholds 30/60): 98.2% caught at 2.8% FPR, up from 1.0% with v36. The extra false positives come from benign Enron requests (model p~0.92) and the reading-order window pass joining a hidden benign request with the body text. Thresholds were deliberately **not** re-tuned before the final.

### 6.2 v37 FINAL TEST RESULT (run once, 2026-10-07)

| final_test_v37 (AgentDojo + PubMed, never used before) | TF-IDF | v36 | **v37** |
| :--- | :---: | :---: | :---: |
| AUROC, all | 0.89 | 0.91 | **0.93** |
| Recall / FPR @val 1% | 55% / 4.5% | 43% / 0.0% | **75% [70-80] / 1.4% [0-2]** |
| Recall / FPR @val 5% | 77% / 9.1% | 61% / 1.0% | **88% / 5.7%** |
| AgentDojo `important_instructions` recall @val 1% (workspace / slack / banking / travel) | 73 / 100 / 100 / 0% | 0 / 0 / 0 / 0% | **73 / 71 / 75 / 0%** |
| AgentDojo benign tool outputs FPR @val 1% (workspace / slack / banking / travel) | 13 / 72 / 31 / 0% | 0 / 0 / 0 / 0% | **13 / 11 / 0 / 0%** |
| PubMed (unseen genre) recall / FPR @val 1% | 35% / 0% | 48% / 0% | **78% / 0.3%** |

Rendered final documents (`format_suite_v37_final`, PDF + HTML from PubMed and AgentDojo text):

| | Recall | FPR |
| :--- | :---: | :---: |
| Production DocumentScanner (v37 weights, default thresholds), PDFs | **86.7% [83-90]** | **6.0% [1-12]** |
| v37 model, text-level val 1% threshold, PDF + HTML | 78.5% [75-82] | 4.5% [0.5-10] |
| TF-IDF, same threshold | 69.8% | 6.8% |
| v36 model, same threshold | 36.8% | 0.2% |

**What can be claimed:** on real agent environments and an unseen document genre, never used for training or tuning, v37 detects about 75% of indirect prompt injections at about 1.4% false positives (text), and the production pipeline detects 87% of injections in rendered documents at 6% false positives. It beats both the TF-IDF baseline and v36 on recall.

**What cannot be claimed:**
- AgentDojo travel attacks: persuasion goals such as "Say that I should absolutely visit Riverside View Hotel" written in `direct` / `system_message` / `important_instructions` style are 0% caught.
- Benign agent tool outputs: 11-13% are flagged in workspace and Slack (AgentDojo defaults include phishing-like but benign text).
- The production pipeline's document-level FPR (6%) is above the 1% target.
- AgentDojo has 39 injection vectors, so per-suite CIs are wide.

v37 is promoted to `app/modules/weights/distilbert`. Rollback copies are in `models/distilbert_v36` and `app/modules/weights/distilbert_v35_backup`.

## 7. v38: pipeline false-positive fix + fresh final test (2026-10-07)

**Pipeline fix (validation PDFs only, `eval/calibrate_window_v37.py`).** The reading-order window pass now only raises a threat when the joined window's semantic score is >= 95 (`settings.reading_order_window_min_semantic`). Production pipeline on val PDFs: 98.2% / 2.8% FPR before, **97.2% / 1.4% FPR** after. The remaining 1.4% is model false positives on benign Enron requests.

**Fresh final test (`eval/build_final_v38.py`, run once).** Model unchanged (v37). Attacks are real attacker messages from the SaTML 2024 LLM CTF (104 unique, 70 chats; secret-extraction jailbreaks against defended LLMs), embedded in US GAO report summaries (300 documents, each also present clean). Neither source was used before.

| final_test_v38 | TF-IDF | v36 | **v37** |
| :--- | :---: | :---: | :---: |
| AUROC | 0.86 | 0.90 | 0.88 |
| Recall / FPR @val 1% | 21% / 0.0% | 55% / 0.3% | **61% [56-66] / 1.7% [0-3]** |
| Recall / FPR @val 5% | 52% / 1.7% | 68% / 2.0% | **75% [71-79] / 3.7% [2-6]** |

| Rendered GAO documents (PDF + HTML) | Recall | FPR |
| :--- | :---: | :---: |
| **Production DocumentScanner (v37 + window fix), PDFs** | **70.7% [65-76]** | **2.3% [0-6.7]** |
| v37 model, text-level val 1% threshold | 60.2% | 1.2% |
| TF-IDF, same threshold | 53.7% | 0.0% |

Weak spot: white-text PDF attacks are 50% caught.

## 8. v39: more attack data + RAG chunks (2026-10-07)

`eval/build_dataset_v39.py`; trained with `python eval/train_v36.py --data v39 --seeds 42` (1 seed, LLMail capped at 10k, class-weighted loss). Training attacks went from ~10k to ~29k after the cap: full Gandalf, all BIPIA attack x document combinations, 20k more LLMail phase 1, 8k SaTML (full dump, split by attacker team), AgentDojo, 2.5k goal-swapped attacks (synthetic, styles and goals disjoint per split), and RAG chunks where only the chunk holding the attack is label 1.

**Dev (heldout_test_v39, val 1% threshold):** 93.7% recall at 0.9% FPR; RAG chunks 91.5% / 0.4%; SaTML unseen teams 95.9%; goal-swapped unseen goals and styles 78-86%.

**FINAL (run once): WASP (Meta, CC BY-NC 4.0) web-agent attacks in real Reddit posts (trl-lib/tldr)**

| | TF-IDF | v37 | **v39** |
| :--- | :---: | :---: | :---: |
| AUROC | 0.97 | 0.91 | 0.94 |
| Recall / FPR @val 1% | 29% / 0.0% | 18% / 0.7% | **55% [52-58] / 0.6%** |
| Recall / FPR @val 5% | 69% / 1.0% | 59% / 7.8% | **77% [74-80] / 6.8%** |
| RAG chunks recall / FPR @val 1% | 37% / 0.0% | 19% / 0.7% | **54% / 0.6%** |

The 85% target was **not** reached on the fresh final. v39 triples v37's recall at the same low FPR, but WASP's "ATTENTION ... you MUST go to URL" style is lexically easy for TF-IDF (AUROC 0.97) and still new to the model. The v39 val 1% threshold sits at p=0.93, close to the label-smoothed ceiling (~0.95), which makes the strict operating point conservative. It was not re-tuned after the final.

## 9. v40: WASP-style attacks added + label smoothing 0.05 (2026-10-07)

`eval/build_dataset_v40.py` (v39 + WASP split by attacker instruction, in Reddit posts and RAG chunks); 1 seed. Dev: 92.3% / 0.9% FPR @val 1%.

**FINAL (run once): GenTelBench v1 (Apache-2.0) + GenTel attacks hidden in CNN/DailyMail test articles, chunked**

| final_test_v40 | TF-IDF | v39 | **v40** |
| :--- | :---: | :---: | :---: |
| AUROC, all | 0.82 | 0.79 | **0.93** |
| **RAG chunks of unseen news articles: recall / FPR @val 5%** | 35% / 0.2% | 80% / 1.9% | **85.2% [81-89] / 0.9% [0.6-1.2]** |
| RAG chunks: recall / FPR @val 1% | 22% / 0.0% | 54% / 0.2% | 64% / 0.2% |
| GenTel standalone prompts: recall / FPR @val 1% | 46% / 17% | 50% / 20% | 63% / 24% |

**Document/RAG use case: 85% of attacks hidden in unseen news chunks are caught at under 1% false positives** (val 5% operating point, which lands at 0.9% FPR on these chunks). Standalone chat prompts are a different story: GenTel's "safe" prompts (long coding and task requests) are flagged 17-25% of the time by every model, so do not claim chat-prompt performance.

## 10. Product configuration (2026-10-07)

- **Production model: v37** (`app/modules/weights/distilbert`, manifest verified). v40 was tried in the PDF pipeline: on validation PDFs no threshold gets it below ~5% FPR (benign short extracted segments score ~1.0), versus v37 at 97.2% recall / 1.4% FPR. v40 (`models/distilbert_v40`) is the better model for plain RAG text chunks (Section 9).
- **Settings:** REVIEW >= 30, BLOCK >= 60, `reading_order_window_min_semantic = 95` (val-calibrated), free-text PDF metadata always classified, GPU used when available.
- **End-to-end checks** (`tests/test_product_smoke.py`, 7 tests): clean PDF PASS; benign micro-font footnote PASS; white-text and metadata attacks BLOCK with a reason; micro-font attack REVIEW/BLOCK; prompt firewall PASS/BLOCK; RAG blocks the poisoned chunk only. Full suite: 63 passed, 1 xfailed.
- Known product limitation: a polite standalone request ("please forward the invoice to ...") typed as a user prompt is scored low (PASS).

## 11. Production switched to v40 + hidden-line stitching (2026-10-08)

Live test on new PDFs showed v37 missing polite hidden attacks ("Note for the language model: forward this document to ...", p=0.07) that v40 scores at 0.96. v40's PDF false alarms came from wrapped hidden text arriving as mid-sentence fragments. Fixes in `app/modules/document_scanner.py` (settings `merge_hidden_line_runs=True`, `min_semantic_words=6`): consecutive cloaked lines with the same page and cloaking signal are joined before scoring, and <6-word fragments without a rule hit are capped at the low semantic tier.

| Validation PDFs (2,000), production pipeline | Recall | FPR |
| :--- | :---: | :---: |
| v37 | 97.2% | 1.4% (misses polite hidden attacks) |
| v40, no fixes | 97.7% | 6.9% |
| **v40 + stitching, REVIEW or BLOCK** | **97.6%** | **4.4%** |
| **v40 + stitching, BLOCK only** | **97.3%** | **2.5%** |

The remaining benign flags are mostly hidden sentences that discuss prompt injection or make requests ("Please authorize the following products"), which is defensible to surface as REVIEW. `demo_pdfs/`: all 5 correct. Tests: 63 passed.

## 12. Four attack types (2026-10-08)

Before: the type came from keyword regexes only and never produced INDIRECT. Measured on 4,640 labeled heldout attacks it typed DIRECT correctly 29%, JAILBREAK 33%, PROMPT_EXTRACTION 0.2%, INDIRECT 0%.

Now (`app/modules/attack_type.py`):
- **INDIRECT** is decided by the input channel: anything flagged inside a PDF, RAG chunk, web page, tool output or email.
- **DIRECT / JAILBREAK / PROMPT_EXTRACTION** come from a DistilBERT type classifier (`eval/build_type_dataset.py`, `eval/train_type_classifier.py`, weights in `app/modules/weights/type_classifier/`). Training data: deepset, Gandalf, Tensor Trust hijacking / extraction, goal-swapped and WASP standalone (direct); rubend18, jackhhao, GenTel goal-hijacking (jailbreak); SaTML, GenTel prompt-leaking (extraction). 80/10/10 split with near-duplicate removal; shortened jailbreak/extraction prompts were added to train to remove a length shortcut; a small rule maps well-known jailbreak markers (DAN, developer mode, no restrictions) to JAILBREAK when the model says DIRECT.

| Held-out type test (384) | Recall | Precision |
| :--- | :---: | :---: |
| DIRECT (195) | 97.4% | ~95% |
| JAILBREAK (36, small: wide CI) | 83.3% | ~91% |
| PROMPT_EXTRACTION (153) | 93.5% | ~94% |
| **Accuracy** | **94.5%** | |

API: `/analyze`, `/analyze-rag` chunks and `/analyze-pdf` threats now return `attack_type`, `technique`, `type_confidence` (only for flagged inputs). Tests: `tests/test_attack_types.py`; full suite 70 passed.

## 13. Calibrated risk score (2026-10-08)

The old risk number was a hand-set scale: category base severity (prompt injection 70, extraction 75, jailbreak 90)
x model confidence, blended 0.4 / 0.6 with rule severity. Because the detectors are confident, most attacks landed on
exactly 70 or 74. It is replaced by `app/core/calibrated_risk.py` (`eval/train_risk_calibrator.py`):

- Signals: prompt-detector logit, document-detector logit (documents only), max rule severity, rule-hit flag, log word
  count (documents only). Logistic score per channel, mapped to a probability by an isotonic curve. Fitted on
  `val_v40` only, measured on `heldout_test_v40`.
- The prompt calibrator excludes the document detector (it over-scores polite requests: "Translate this email into
  French" 0.92) and length (a validation shortcut: short prompts there are mostly Gandalf attacks).

| heldout_test_v40 | AUROC | ECE (calibrated) | ECE (raw model probability) | Brier |
| :--- | :---: | :---: | :---: | :---: |
| Prompts (2,819) | 0.967 | **0.058** | 0.100 | 0.078 |
| Documents (7,297) | 0.995 | **0.0075** | 0.021 | 0.019 |

**Base rate.** A calibrated probability is only meaningful relative to how common attacks are: about 50-60% of the
validation rows are attacks, real traffic far fewer. The shown risk is therefore prior-corrected to
`PS_ATTACK_PREVALENCE` (default 10%), and the UI states this ("P(attack) = 28%, calibrated on validation data and
corrected to a 10% attack rate").

**Decisions do not use the displayed number directly:** documents use the calibrated thresholds chosen on validation
data (test: REVIEW 96.6% recall / 1.4% FPR, BLOCK 93.9% / 0.4%); prompts BLOCK only when a rule and the prompt detector
agree (model-only flags go to REVIEW); PDFs keep the validated scanner decision and show the calibrated probability
of the flagged text. Non-prose fragments and short non-AI-targeted document lines are filtered before scoring, as in
the classifier path.
