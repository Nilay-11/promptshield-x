"""
PromptShield X - Phase 3.6 training on the cleaned v36 datasets.

Trains DistilBERT on train_v36.json, selects the epoch by val loss on val_v36.json and
calibrates thresholds on val only. Writes to models/distilbert_v36/ and does NOT touch the
production weights unless --promote is passed.

Usage:
    python eval/train_v36.py                 # seed 42
    python eval/train_v36.py --seeds 42 123 456
    python eval/train_v36.py --promote       # also copy best seed to app/modules/weights/distilbert
"""

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("eval/data")
OUTPUT_DIR = Path("models/distilbert_v36")
PROD_DIR = Path("app/modules/weights/distilbert")
MANIFEST_PATH = Path("app/modules/weights/manifest.json")

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 4e-5
MAX_LENGTH = 512  # matches the production sliding window; doc chunks reach ~375 tokens
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(rows, tokenizer):
    enc = tokenizer([r["text"] for r in rows], truncation=True, max_length=MAX_LENGTH)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "labels": rows[i]["label"]} for i in range(len(rows))]


def length_bucketed(features, batch_size, rng):
    """Shuffle, then sort within chunks of 50 batches so padding stays small."""
    idx = list(range(len(features)))
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: len(features[i]["input_ids"]))
        batches += [part[j : j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(model, features, collator):
    model.eval()
    order = sorted(range(len(features)), key=lambda i: len(features[i]["input_ids"]))
    probs = np.zeros(len(features))
    for s in range(0, len(order), 64):
        ids = order[s : s + 64]
        batch = collator([{k: v for k, v in features[i].items() if k != "labels"} for i in ids])
        logits = model(**{k: v.to(DEVICE) for k, v in batch.items()}).logits
        probs[ids] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def threshold_at_fpr(labels, probs, target):
    neg = np.sort(probs[labels == 0])
    k = min(int(np.floor((1.0 - target) * len(neg))), len(neg) - 1)
    return float(neg[k]) + 1e-6


def val_metrics(labels, probs):
    eps = 1e-7
    loss = -np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps))
    out = {"loss": float(loss), "auroc": float(roc_auc_score(labels, probs))}
    for name, target in (("fpr1", 0.01), ("fpr5", 0.05)):
        tau = threshold_at_fpr(labels, probs, target)
        out[f"tau_{name}"] = tau
        out[f"recall_at_{name}"] = float(np.mean(probs[labels == 1] >= tau))
    return out


def train_seed(seed, train_f, val_f, val_labels, tokenizer, collator):
    print(f"\n--- Seed {seed} ---", flush=True)
    set_seed(seed)
    rng = random.Random(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    steps_per_epoch = (len(train_f) + BATCH_SIZE - 1) // BATCH_SIZE
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps_per_epoch * EPOCHS),
                                                steps_per_epoch * EPOCHS)
    best, best_state = None, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        batches = length_bucketed(train_f, BATCH_SIZE, rng)
        for step, ids in enumerate(batches, 1):
            batch = {k: v.to(DEVICE) for k, v in collator([train_f[i] for i in ids]).items()}
            loss = model(**batch).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step}/{len(batches)} loss {running / step:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        m = val_metrics(val_labels, predict(model, val_f, collator))
        print(f"Seed {seed} epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        if best is None or m["loss"] < best["loss"]:
            best, best_state = m, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def dir_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob("**/*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--promote", action="store_true", help="copy best seed into production weights")
    args = ap.parse_args()

    train = json.loads((DATA_DIR / "train_v36.json").read_text(encoding="utf-8"))
    val = json.loads((DATA_DIR / "val_v36.json").read_text(encoding="utf-8"))
    print(f"train_v36: {len(train)} | val_v36: {len(val)} | device {DEVICE}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer)
    # An embedded attack past MAX_LENGTH would be truncated away, leaving a "label 1" row with
    # no attack in view. Drop those rows (benign rows stay benign when truncated).
    too_long = lambda r: r.get("doc_injected") and len(tokenizer(r["text"])["input_ids"]) > MAX_LENGTH
    before = len(train)
    train = [r for r in train if not too_long(r)]
    print(f"dropped {before - len(train)} doc-injected rows longer than {MAX_LENGTH} tokens", flush=True)
    train_f, val_f = encode(train, tokenizer), encode(val, tokenizer)
    val_labels = np.array([r["label"] for r in val])

    results, best_seed, best_model = {}, None, None
    for seed in args.seeds:
        model, m = train_seed(seed, train_f, val_f, val_labels, tokenizer, collator)
        results[seed] = m
        if best_seed is None or m["loss"] < results[best_seed]["loss"]:
            best_seed, best_model = seed, model

    summary = {k: {"mean": float(np.mean([results[s][k] for s in args.seeds])),
                   "std": float(np.std([results[s][k] for s in args.seeds]))}
               for k in results[best_seed]}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    meta = {
        "version": "3.6.0-clean-data",
        "dataset": "train_v36 / val_v36",
        "architecture": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "best_seed": best_seed,
        "seeds": {str(s): r for s, r in results.items()},
        "summary": summary,
        "validation_metrics": results[best_seed],
    }
    (OUTPUT_DIR / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved seed {best_seed} to {OUTPUT_DIR}")

    if args.promote:
        if PROD_DIR.exists():
            shutil.rmtree(PROD_DIR)
        shutil.copytree(OUTPUT_DIR, PROD_DIR)
        manifest = {"distilbert": {**meta, "sha256": dir_sha256(PROD_DIR)}}
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Promoted to {PROD_DIR} and updated {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
