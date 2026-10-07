"""
PromptShield X - attack-TYPE dataset (direct / jailbreak / prompt extraction).

INDIRECT is decided by the input channel (PDF, RAG chunk, tool output, email), not by text: an attack that
arrives inside third-party content is indirect by definition. This dataset trains the 3-way text classifier
that names the technique of an attack the detector has already flagged.

Type labels come from what each source contains:
  DIRECT            deepset injections, Gandalf (Lakera), Tensor Trust hijacking, goal-swapped standalone
                    attacks, WASP standalone instructions
  JAILBREAK         rubend18 jailbreak prompts, jackhhao jailbreaks, GenTelBench goal-hijacking (DAN /
                    developer-mode style), Lakera-style persona prompts in deepset are NOT relabeled
  PROMPT_EXTRACTION SaTML CTF secret-extraction turns, Tensor Trust extraction attacks, GenTelBench prompt-leaking
Split 80/10/10 by normalized text with exact and near-duplicate removal across splits.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v36 import DATA_DIR, RAW_DIR, ngrams, norm  # noqa: E402

TYPES = ["DIRECT", "JAILBREAK", "PROMPT_EXTRACTION"]
SEED = 7
CAP = 4000  # per type, keeps classes balanced


def collect():
    rows = []
    add = lambda t, typ, src: rows.append({"text": t.strip(), "type": typ, "source": src}) if 15 <= len(t.strip()) <= 2000 else None

    ds = load_dataset("deepset/prompt-injections")
    for sp in ("train", "test"):
        for r in ds[sp]:
            if int(r["label"]) == 1:
                add(r["text"], "DIRECT", "deepset")
    g = load_dataset("Lakera/gandalf_ignore_instructions")
    for sp in g:
        for r in g[sp]:
            add(r["text"], "DIRECT", "gandalf")
    for f, typ in (("benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl", "DIRECT"),
                   ("benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl", "PROMPT_EXTRACTION")):
        with open(hf_hub_download("qxcv/tensor-trust", f, repo_type="dataset"), encoding="utf-8") as fh:
            for line in fh:
                add(json.loads(line).get("attack", ""), typ, "tensor_trust")
    for name in ("train_v40", "val_v40", "heldout_test_v40"):
        for r in json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8")):
            if r["label"] != 1:
                continue
            s = r["source"]
            if s.startswith(("goalswap_standalone", "wasp_standalone")):
                add(r["text"], "DIRECT", s.split("_")[0])
            elif s.startswith("rubend18"):
                add(r["text"], "JAILBREAK", "rubend18")
            elif s.startswith("jackhhao"):
                add(r["text"], "JAILBREAK", "jackhhao")
            elif s.startswith("satml"):
                add(r["text"], "PROMPT_EXTRACTION", "satml")
    for f, typ in (("goal_hijacking_attack_dataset.parquet", "JAILBREAK"),
                   ("prompt_leaking_attack_dataset.parquet", "PROMPT_EXTRACTION")):
        d = pd.read_parquet(hf_hub_download("GenTelLab/gentelbench-v1", f, repo_type="dataset"))
        for t, lab in zip(d.text, d.label):
            if int(lab) == 1:
                add(str(t), typ, "gentel")
    return rows


def main():
    rng = random.Random(SEED)
    rows = collect()
    by_key = {}
    conflicts = set()
    for r in rows:
        k = norm(r["text"])
        if k in by_key and by_key[k]["type"] != r["type"]:
            conflicts.add(k)
        by_key.setdefault(k, r)
    uniq = [r for k, r in by_key.items() if k not in conflicts]
    out = []
    for typ in TYPES:
        xs = [r for r in uniq if r["type"] == typ]
        rng.shuffle(xs)
        # balance by capping each source inside the type, then the type
        per_src = Counter()
        capped = []
        for r in xs:
            if per_src[r["source"]] < CAP // 2:
                per_src[r["source"]] += 1
                capped.append(r)
        out += capped[:CAP]
    rng.shuffle(out)
    n = len(out)
    splits = {"train": out[: int(.8 * n)], "val": out[int(.8 * n): int(.9 * n)], "test": out[int(.9 * n):]}
    seen = set()
    for sp in ("train", "val", "test"):  # near-duplicate removal against earlier splits
        kept = []
        for r in splits[sp]:
            g = ngrams(r["text"])
            if sp != "train" and g and len(g & seen) / len(g) > 0.5:
                continue
            kept.append(r)
        splits[sp] = kept
        for r in kept:
            seen |= ngrams(r["text"])
        (DATA_DIR / f"types_{sp}.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False), encoding="utf-8")
        print(sp, len(kept), dict(Counter(r["type"] for r in kept)), dict(Counter(r["source"] for r in kept)))
    print("label conflicts dropped:", len(conflicts))


if __name__ == "__main__":
    main()
