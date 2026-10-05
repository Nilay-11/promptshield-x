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
