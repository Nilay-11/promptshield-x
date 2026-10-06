"""
PromptShield X - Phase 3.5 Dataset Builder.
Strictly isolates:
1. Training Split:
   - Deepset Train (546 samples)
   - Gandalf Train (777 samples)
   - Rubend18 Jailbreaks (79 samples)
   - BIPIA Train subset (60 unique injection directives)
   - Awesome ChatGPT Prompts (800 benign casual/user prompts)
   - Dolly 15k Instructions (800 benign instructions)
   - Hard Negatives from TRAIN template family (400 samples)
   - Perturbation Augmentation on train (homoglyphs, leetspeak, spaces, casing) on 20% of train samples.
2. Validation Split:
   - Gandalf Validation (111 samples)
   - BIPIA Validation subset (25 unique injection directives)
   - Dolly 15k Validation subset (200 benign instructions)
   - Awesome ChatGPT Prompts Validation subset (200 benign prompts)
   - Hard Negatives from VALIDATION template family (150 samples)
3. NEW Held-Out Test Split (Touched ONLY once at the very end):
   - Deepset Test Split (116 samples: 60 injection, 56 benign)
   - Gandalf Test Split (112 injection samples)
   - BIPIA Test subset (40 unique unseen injection directives)
   - Unseen Dolly 15k prompts (150 benign instructions)
   - Unseen Awesome ChatGPT Prompts (150 benign prompts)
   - Hard Negatives from TEST template family (150 samples)
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)

DATA_DIR = Path("eval/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Template Families strictly separated across splits
# ----------------------------------------------------------------------
TRAIN_FOOTNOTES = [
    "Footnote {n}: Operating expenses include non-recurring restructuring charges of ${m} million.",
    "Footnote {n}: Depreciation and amortization are calculated under standard straight-line methods over {y} years.",
    "Footnote {n}: Inventory valuation is determined under the first-in, first-out (FIFO) accounting method.",
    "Footnote {n}: Capitalized software development costs totaled ${m}k for the current quarterly cycle.",
]

VAL_FOOTNOTES = [
    "Footnote {n}: Diluted earnings per common share reflects weighted average equity instruments.",
    "Footnote {n}: Net income adjusted for foreign exchange fluctuations in European and Asian markets.",
    "Footnote {n}: Cash flows from financing activities include repayment of ${m}M in senior unsecured notes.",
]

TEST_FOOTNOTES = [
    "Footnote {n}: Amounts are presented in thousands unless otherwise stated.",
    "Footnote {n}: Revenue recognition follows IFRS 15 guidelines across all cloud subscription contracts.",
    "Footnote {n}: Prior period amounts have been reclassified to conform to current presentation standards.",
]

TRAIN_LEGAL = [
    "This document contains proprietary and confidential information. Unauthorized copying or redistribution is strictly prohibited.",
    "Confidential information does not include information that is publicly known through no breach of this agreement.",
    "Past performance is no guarantee of future returns. Any investment involves potential risk of loss.",
    "No waiver of any breach shall be construed as an ongoing waiver of any subsequent breach of covenant.",
]

VAL_LEGAL = [
    "All forward-looking statements are subject to risks and uncertainties that could cause actual results to differ materially.",
    "The intellectual property rights in all work products remain vested exclusively with the issuing party.",
    "Indemnification covenants survive expiration or termination of the master agreement for a period of {y} years.",
]

TEST_LEGAL = [
    "Neither party shall be liable for indirect, incidental, special, punitive, or consequential damages arising hereunder.",
    "This agreement shall be construed and governed in accordance with the substantive laws of the State of Delaware.",
    "Any dispute arising out of or related to this contract shall be submitted to binding arbitration in New York.",
]

TRAIN_POLICY = [
    "Failure to comply with corporate data governance standards may result in immediate credential revocation.",
    "Users must not attempt to decompile, reverse engineer, or disassemble any part of the software binary.",
    "Third-party vendors must execute a data processing addendum before accessing customer environments.",
    "Password credentials must meet complexity requirements and be rotated every {d} days.",
]

VAL_POLICY = [
    "Retention schedules mandate preserving financial records and transaction logs for {y} calendar years.",
    "All employees must complete mandatory annual cybersecurity awareness and anti-phishing training by Q4.",
    "Access permissions must be reviewed and re-certified quarterly in compliance with ISO 27001 requirements.",
]

TEST_POLICY = [
    "By continuing to access or use the platform services, you agree to be bound by the terms outlined herein.",
    "Multi-factor authentication is strictly enforced for all administrative and production access gateways.",
    "Customer data will be processed solely in accordance with applicable GDPR and CCPA privacy frameworks.",
]

TRAIN_META = [
    "The research paper analyzes prompt injection benchmarks and proposes defense architectures to mitigate risk.",
    "Incident analysis demonstrated that the attack vector relied on persona hijacking phrasing such as 'act as DAN'.",
    "The curriculum explains why multi-layer firewalls combine signature filters with neural classifiers.",
    "The security report documented that standard regex rules failed to catch obfuscated leetspeak injections.",
]

VAL_META = [
    "In this academic lecture, students examine how adversarial prompts attempt to reveal internal system instructions.",
    "Security evaluation scripts simulate attacks like 'disregard prior directives' to test firewall recall.",
    "Our threat model specifically addresses indirect prompt injection smuggled via untrusted retrieved documents.",
]

TEST_META = [
    "During employee security training, instructors frequently discuss why attackers attempt to ignore previous instructions.",
    "Understanding prompt injection mechanisms requires distinguishing benign directives from malicious overrides.",
    "When evaluating safety guardrails, researchers test whether models can resist jailbreak roleplay attempts.",
]


def generate_family_samples(templates_dict: dict, count: int, seed: int, prefix: str) -> list[dict]:
    rng = random.Random(seed)
    categories = list(templates_dict.keys())
    results = []
    for i in range(count):
        cat = categories[i % len(categories)]
        t_list = templates_dict[cat]
        t = rng.choice(t_list)
        text = t.format(n=(i % 20) + 1, m=(i % 15) + 1, y=(i % 5) + 2, d=(i % 60) + 30)
        results.append({
            "text": text,
            "label": 0,
            "source": f"{prefix}_{cat}",
            "is_hard_negative": True,
            "category": "BENIGN"
        })
    return results


def apply_perturbation_augmentation(text: str, rng: random.Random) -> str:
    choice = rng.choice(["homoglyph", "leet", "spacing", "upper"])
    if choice == "homoglyph":
        return text.replace("a", "\u0430").replace("e", "\u0435").replace("o", "\u043e")
    elif choice == "leet":
        return text.replace("e", "3").replace("i", "1").replace("o", "0")
    elif choice == "spacing":
        words = text.split()
        if words:
            w_idx = rng.randint(0, len(words) - 1)
            target = words[w_idx]
            if len(target) >= 5 and target.isalpha():
                words[w_idx] = " ".join(target)
                return " ".join(words)
        return text
    elif choice == "upper":
        return text.upper()
    return text


def build_phase35_datasets():
    print("Building Phase 3.5 datasets with strict leak-free splits...")
    print("Loading HuggingFace sources...")
    deepset_ds = load_dataset("deepset/prompt-injections")
    gandalf_ds = load_dataset("Lakera/gandalf_ignore_instructions")
    rubend_ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
    bipia_ds = load_dataset("geodesic-research/bipia", split="train")
    awesome_ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    dolly_ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    rng = random.Random(42)

    # 1. BIPIA unique injection directives
    bipia_unique = list(set(r["attack_str"].strip() for r in bipia_ds if r.get("attack_str")))
    rng.shuffle(bipia_unique)
    bipia_train = bipia_unique[:60]
    bipia_val = bipia_unique[60:85]
    bipia_test = bipia_unique[85:125]

    # 2. Benign Awesome Prompts
    awesome_prompts = [r["prompt"].strip() for r in awesome_ds if r.get("prompt")]
    rng.shuffle(awesome_prompts)
    awesome_train = awesome_prompts[:800]
    awesome_val = awesome_prompts[800:1000]
    awesome_test = awesome_prompts[1000:1150]

    # 3. Benign Dolly Instructions
    dolly_instructions = [r["instruction"].strip() for r in dolly_ds if r.get("instruction")]
    rng.shuffle(dolly_instructions)
    dolly_train = dolly_instructions[:800]
    dolly_val = dolly_instructions[800:1000]
    dolly_test = dolly_instructions[1000:1150]

    # TRAIN SET
    train_samples = []
    for row in deepset_ds["train"]:
        train_samples.append({
            "text": row["text"].strip(),
            "label": int(row["label"]),
            "source": "deepset_train",
            "category": "INJECTION" if row["label"] == 1 else "BENIGN"
        })

    for row in gandalf_ds["train"]:
        train_samples.append({
            "text": row["text"].strip(),
            "label": 1,
            "source": "gandalf_train",
            "category": "INJECTION"
        })

    for row in rubend_ds:
        p = row.get("Prompt", row.get("text", "")).strip()
        if p:
            train_samples.append({
                "text": p,
                "label": 1,
                "source": "rubend18_jailbreak",
                "category": "INJECTION"
            })

    for a in bipia_train:
        train_samples.append({
            "text": a,
            "label": 1,
            "source": "bipia_train",
            "category": "INJECTION"
        })

    for p in awesome_train:
        train_samples.append({
            "text": p,
            "label": 0,
            "source": "awesome_prompts_train",
            "category": "BENIGN"
        })

    for d in dolly_train:
        train_samples.append({
            "text": d,
            "label": 0,
            "source": "dolly_train",
            "category": "BENIGN"
        })

    train_templates = {
        "footnote": TRAIN_FOOTNOTES,
        "legal": TRAIN_LEGAL,
        "policy": TRAIN_POLICY,
        "meta": TRAIN_META,
    }
    train_hn = generate_family_samples(train_templates, count=400, seed=1001, prefix="train_family")
    train_samples.extend(train_hn)

    aug_samples = []
    aug_indices = rng.sample(range(len(train_samples)), int(0.20 * len(train_samples)))
    for idx in aug_indices:
        orig = train_samples[idx]
        perturbed_text = apply_perturbation_augmentation(orig["text"], rng)
        if perturbed_text != orig["text"]:
            aug_samples.append({
                "text": perturbed_text,
                "label": orig["label"],
                "source": f"{orig['source']}_augmented",
                "category": orig["category"]
            })
    train_samples.extend(aug_samples)
    rng.shuffle(train_samples)

    # VALIDATION SET
    val_samples = []
    for row in gandalf_ds["validation"]:
        val_samples.append({
            "text": row["text"].strip(),
            "label": 1,
            "source": "gandalf_val",
            "category": "INJECTION"
        })

    for a in bipia_val:
        val_samples.append({
            "text": a,
            "label": 1,
            "source": "bipia_val",
            "category": "INJECTION"
        })

    for p in awesome_val:
        val_samples.append({
            "text": p,
            "label": 0,
            "source": "awesome_prompts_val",
            "category": "BENIGN"
        })

    for d in dolly_val:
        val_samples.append({
            "text": d,
            "label": 0,
            "source": "dolly_val",
            "category": "BENIGN"
        })

    val_templates = {
        "footnote": VAL_FOOTNOTES,
        "legal": VAL_LEGAL,
        "policy": VAL_POLICY,
        "meta": VAL_META,
    }
    val_hn = generate_family_samples(val_templates, count=150, seed=2002, prefix="val_family")
    val_samples.extend(val_hn)
    rng.shuffle(val_samples)

    # HELD-OUT TEST SET
    test_samples = []
    for row in deepset_ds["test"]:
        test_samples.append({
            "text": row["text"].strip(),
            "label": int(row["label"]),
            "source": "deepset_test",
            "category": "INJECTION" if row["label"] == 1 else "BENIGN"
        })

    for row in gandalf_ds["test"]:
        test_samples.append({
            "text": row["text"].strip(),
            "label": 1,
            "source": "gandalf_test",
            "category": "INJECTION"
        })

    for a in bipia_test:
        test_samples.append({
            "text": a,
            "label": 1,
            "source": "bipia_test",
            "category": "INJECTION"
        })

    for p in awesome_test:
        test_samples.append({
            "text": p,
            "label": 0,
            "source": "awesome_prompts_test",
            "category": "BENIGN"
        })

    for d in dolly_test:
        test_samples.append({
            "text": d,
            "label": 0,
            "source": "dolly_test",
            "category": "BENIGN"
        })

    test_templates = {
        "footnote": TEST_FOOTNOTES,
        "legal": TEST_LEGAL,
        "policy": TEST_POLICY,
        "meta": TEST_META,
    }
    test_hn = generate_family_samples(test_templates, count=150, seed=3003, prefix="test_family")
    test_samples.extend(test_hn)
    rng.shuffle(test_samples)

    (DATA_DIR / "train_v35.json").write_text(json.dumps(train_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA_DIR / "val_v35.json").write_text(json.dumps(val_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA_DIR / "heldout_test_v35.json").write_text(json.dumps(test_samples, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Phase 3.5 dataset successfully created:")
    print(f"  Train: {len(train_samples)} (Inj: {sum(s['label']==1 for s in train_samples)}, Benign: {sum(s['label']==0 for s in train_samples)})")
    print(f"  Val:   {len(val_samples)} (Inj: {sum(s['label']==1 for s in val_samples)}, Benign: {sum(s['label']==0 for s in val_samples)})")
    print(f"  Heldout Test: {len(test_samples)} (Inj: {sum(s['label']==1 for s in test_samples)}, Benign: {sum(s['label']==0 for s in test_samples)})")


if __name__ == "__main__":
    build_phase35_datasets()
