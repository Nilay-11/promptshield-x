"""
PromptShield X - fresh final test v38 (after the v37 final was spent).

Model and training data are unchanged from v37 (train/val/dev splits are copied as *_v38). Only the
final test is new, built from two sources never used anywhere in this project:
  * Attacks: real attacker messages from the SaTML 2024 LLM CTF (ethz-spylab/ctf-satml24, MIT),
    the 50-chat sample plus the 50 chats that successfully extracted the secret. Junk turns are
    dropped. group = chat id.
  * Benign documents: US GAO report summaries (ccdv/govreport-summarization, test split), an unseen
    genre. Every document appears clean (label 0) and with one SaTML attack inserted at a sentence
    boundary (label 1), so the genre carries no label signal.
format_suite_v38_final.json renders the same documents as PDF / HTML with visible and hidden attacks
plus benign twins hidden the same way. Run once:
    python eval/evaluate_v36.py --data v38 --final
    python eval/evaluate_documents_v36.py --data v38 --final
"""

import json
import random
import re
import shutil
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v36 import (DATA_DIR, RAW_DIR, build_format_specs, embed_attack, fetch, gid, norm, row,  # noqa: E402
                               summarize, write_split)

SEED = 38
SATML_URL = "https://huggingface.co/datasets/ethz-spylab/ctf-satml24/resolve/main/{}"


def satml_attacks():
    out = {}
    for f in ("chat.50.json", "chat.was_successful_secret_extraction.50.json"):
        chats = json.loads(fetch(SATML_URL.format(f), f"satml_{f}").read_text(encoding="utf-8"))
        for c in chats:
            for m in c.get("history", []):
                t = (m.get("content") or "").strip()
                if m.get("role") != "user" or not (30 <= len(t) <= 1500):
                    continue
                letters = sum(ch.isalpha() for ch in t)
                if letters / len(t) < 0.5 or len(set(t.split())) < 5:  # drop "aaaa..." style junk
                    continue
                out[norm(t)] = (t, f"satml_{c['_id']}")
    return list(out.values())


def govreport(rng, n=300):
    ds = load_dataset("ccdv/govreport-summarization", split="test", streaming=True)
    docs = []
    for r in ds:
        s = re.sub(r"\s+", " ", r["summary"]).strip()
        if 400 <= len(s) <= 4000:
            docs.append(s)
        if len(docs) >= n:
            break
    rng.shuffle(docs)
    return docs[:n]


def main():
    rng = random.Random(SEED)
    for name in ("train", "val", "heldout_test"):
        shutil.copyfile(DATA_DIR / f"{name}_v37.json", DATA_DIR / f"{name}_v38.json")
    for s in ("val", "heldout"):
        shutil.copyfile(DATA_DIR / f"format_suite_v37_{s}.json", DATA_DIR / f"format_suite_v38_{s}.json")

    attacks = satml_attacks()
    docs = govreport(rng)
    print(f"SaTML attacks: {len(attacks)} unique from {len({g for _, g in attacks})} chats | GovReport docs: {len(docs)}")

    # never-seen check against every earlier split
    seen = set()
    for f in ("train_v37", "val_v37", "heldout_test_v37", "final_test_v37", "final_test_v36"):
        seen |= {norm(r["text"]) for r in json.loads((DATA_DIR / f"{f}.json").read_text(encoding="utf-8"))}
    attacks = [(t, g) for t, g in attacks if norm(t) not in seen]

    rows = []
    for d in docs:
        g = f"gov_{gid(norm(d))}"
        rows.append(row(d, 0, "govreport_benign", group=g))
        atk, chat = rng.choice(attacks)
        rows.append(row(embed_attack(d, atk, rng), 1, "govreport_satml_attacked", group=g,
                        doc_injected=True, attack_group=chat))
    rows += [row(t, 1, "satml_attack_standalone", group=g) for t, g in attacks]
    rng.shuffle(rows)
    write_split("final_test_v38", rows)
    report = {"final_test_v38": summarize("final_test_v38", rows), "satml_attacks": len(attacks)}

    specs = build_format_specs([(d, "govreport", False) for d in docs],
                               [t for t, _ in attacks if len(t) <= 1200], rng, n_docs=60)
    (DATA_DIR / "format_suite_v38_final.json").write_text(json.dumps(specs, indent=2, ensure_ascii=False), encoding="utf-8")
    report["format_suite_v38_final"] = len(specs)
    (DATA_DIR / "dataset_report_v38.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"format_suite_v38_final: {len(specs)} rendered specs")


if __name__ == "__main__":
    main()
