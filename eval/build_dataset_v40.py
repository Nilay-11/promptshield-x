"""
PromptShield X - v40: add the attack styles v39 missed + fresh final test.

Train/val/test = v39 plus WASP (Meta, CC BY-NC 4.0; its v39 final is spent) web-agent attacks in all
4 injection formats x 6 user goals, split by attacker instruction (14 train / 3 val / 4 test), both
standalone and inside real Reddit posts and RAG chunks (Reddit train split, never the final posts).

final_test_v40 (touch once; never used anywhere):
  * GenTelBench v1 (GenTelLab, Apache-2.0): goal-hijacking and prompt-leaking attacks WITH their
    matched safe prompts (label 0), standalone.
  * The same GenTel attacks hidden inside CNN/DailyMail test articles, chunked RAG-style, plus the
    clean chunks of the same articles.
"""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset_v36 as v36  # noqa: E402
import build_dataset_v39 as v39  # noqa: E402
from build_dataset_v36 import DATA_DIR, dedupe, gid, label_conflicts_across, norm, row, summarize, write_split  # noqa: E402

SEED = 40


def wasp_by_instruction(rng):
    attacks = v39.wasp_attacks()  # (text, group per instruction)
    groups = sorted({g for _, g in attacks})
    parts = dict(zip(("train", "val", "test"), v36.split_list(groups, [0.65, 0.15, 0.2], rng)))
    return {sp: [t for t, g in attacks if g in set(gs)] for sp, gs in parts.items()}


def reddit_train_posts(rng, n=1500):
    ds = load_dataset("trl-lib/tldr", split="train", streaming=True)
    posts = []
    for r in ds:
        m = re.search(r"POST:\s*(.*?)\s*(TL;DR:)?\s*$", r.get("prompt") or "", flags=re.S)
        t = re.sub(r"\s+", " ", m.group(1) if m else "").strip()
        if 300 <= len(t) <= 2500:
            posts.append(t)
        if len(posts) >= n:
            break
    rng.shuffle(posts)
    return posts


def gentel():
    out = []
    for f, kind in (("goal_hijacking_attack_dataset.parquet", "goal_hijacking"),
                    ("prompt_leaking_attack_dataset.parquet", "prompt_leaking")):
        d = pd.read_parquet(hf_hub_download("GenTelLab/gentelbench-v1", f, repo_type="dataset"))
        for t, lab in zip(d.text, d.label):
            t = str(t).strip()
            if 15 <= len(t) <= 3000:
                out.append((t, int(lab), kind))
    return out


def cnn_articles(rng, n=400):
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
    arts = []
    for r in ds:
        a = re.sub(r"\s+", " ", r["article"]).strip()
        if 1500 <= len(a) <= 6000:
            arts.append(a)
        if len(arts) >= n:
            break
    rng.shuffle(arts)
    return arts


def build():
    rng = random.Random(SEED)
    base = {n: json.loads((DATA_DIR / f"{n}_v39.json").read_text(encoding="utf-8")) for n in ("train", "val", "heldout_test")}
    wasp = wasp_by_instruction(rng)
    posts = reddit_train_posts(rng)
    k1, k2 = int(0.7 * len(posts)), int(0.85 * len(posts))
    post_split = {"train": posts[:k1], "val": posts[k1:k2], "test": posts[k2:]}
    add = {"train": [], "val": [], "test": []}
    for sp in add:
        add[sp] += [row(a, 1, f"wasp_standalone_{sp}") for a in wasp[sp]]
        for p in post_split[sp]:
            g = f"reddit_{sp}_{gid(p)}"
            if rng.random() < 0.5:
                atk = rng.choice(wasp[sp])
                add[sp].append(row(v36.embed_attack(p, atk, rng), 1, f"reddit_wasp_{sp}", group=g, doc_injected=True))
            else:
                add[sp].append(row(p, 0, f"reddit_benign_{sp}", group=g))
        add[sp] += v39.chunked_rows(post_split[sp], wasp[sp], rng, {"train": 600, "val": 120, "test": 120}[sp], sp, "wasp")
    splits = {}
    for sp, name in (("train", "train"), ("val", "val"), ("test", "heldout_test")):
        rows = base[name] + add[sp]
        rng.shuffle(rows)
        splits[sp] = rows

    print("Building final (GenTelBench + CNN/DailyMail)...", flush=True)
    gt = gentel()
    attacks = [t for t, lab, _ in gt if lab == 1 and len(t) <= 1200]
    final = [row(t, lab, f"gentel_{kind}_{'attack' if lab else 'safe'}") for t, lab, kind in gt]
    arts = cnn_articles(rng)
    final += v39.chunked_rows(arts, attacks, rng, 300, "final", "gentel_cnn")
    for a in arts[:100]:  # whole clean articles as extra chunks
        for s, e in v39.chunk_text(a):
            final.append(row(a[s:e].strip(), 0, "rag_chunk_clean_cnn_only_final", group=f"cnn_{gid(a)}"))
    rng.shuffle(final)

    deduped, removed, _ = dedupe([("train", splits["train"]), ("val", splits["val"]),
                                  ("test", splits["test"]), ("final", final)])
    names = {"train": "train_v40", "val": "val_v40", "test": "heldout_test_v40", "final": "final_test_v40"}
    report = {"seed": SEED, "base": "v39", "removed": {f"{k[0]}/{k[1]}": v for k, v in sorted(removed.items())}}
    for key, fname in names.items():
        write_split(fname, deduped[key])
        report[fname] = summarize(fname, deduped[key])
    report["cross_split_label_conflicts"] = label_conflicts_across(deduped)
    print("Removed:", report["removed"], "| conflicts:", report["cross_split_label_conflicts"])
    (DATA_DIR / "dataset_report_v40.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    build()
