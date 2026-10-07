"""
PromptShield X - train the attack-TYPE classifier (DIRECT / JAILBREAK / PROMPT_EXTRACTION).

Fine-tunes DistilBERT on eval/data/types_{train,val}.json, picks the epoch by val macro-F1, reports a
confusion matrix on types_test.json and saves to app/modules/weights/type_classifier/.
INDIRECT is assigned by input channel in the app, not by this model.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

DATA = Path("eval/data")
OUT = Path("app/modules/weights/type_classifier")
TYPES = ["DIRECT", "JAILBREAK", "PROMPT_EXTRACTION"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, LR, MAXLEN = 3, 16, 4e-5, 256


def load(sp):
    rows = json.loads((DATA / f"types_{sp}.json").read_text(encoding="utf-8"))
    return [r["text"] for r in rows], np.array([TYPES.index(r["type"]) for r in rows])


@torch.no_grad()
def predict(model, tok, texts):
    model.eval()
    out = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = load("train"), load("val"), load("test")
    # Length shortcut fix: training jailbreaks are mostly long, so short ones ("you are DAN, no restrictions")
    # were typed DIRECT. Add the first 1-2 sentences of every training jailbreak / extraction prompt.
    import re as _re
    extra_x, extra_y = [], []
    for t, y in zip(tr_x, tr_y):
        if TYPES[y] in ("JAILBREAK", "PROMPT_EXTRACTION") and len(t) > 250:
            sents = _re.split(r"(?<=[.!?])\s+", t)
            short = " ".join(sents[:2])[:300]
            if len(short) >= 30:
                extra_x.append(short); extra_y.append(y)
    tr_x, tr_y = tr_x + extra_x, np.concatenate([tr_y, np.array(extra_y, dtype=tr_y.dtype)])
    print(f"added {len(extra_x)} shortened jailbreak/extraction prompts", flush=True)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3, id2label=dict(enumerate(TYPES)), label2id={t: i for i, t in enumerate(TYPES)}).to(DEVICE)
    enc = tok(tr_x, truncation=True, max_length=MAXLEN)
    feats = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i], "labels": int(tr_y[i])} for i in range(len(tr_x))]
    coll = DataCollatorWithPadding(tok)
    counts = np.bincount(tr_y, minlength=3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(tr_y) / (3 * counts), dtype=torch.float, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = (len(feats) + BATCH - 1) // BATCH * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(len(feats))
        for s in range(0, len(idx), BATCH):
            batch = {k: v.to(DEVICE) for k, v in coll([feats[i] for i in idx[s:s + BATCH]]).items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = loss_fn(model(**batch).logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        f1 = f1_score(va_y, predict(model, tok, va_x).argmax(1), average="macro")
        print(f"epoch {ep + 1}: val macro-F1 {f1:.4f}", flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = predict(model, tok, te_x).argmax(1)
    report = classification_report(te_y, pred, target_names=TYPES, output_dict=True, zero_division=0)
    cm = confusion_matrix(te_y, pred, labels=[0, 1, 2]).tolist()
    print(classification_report(te_y, pred, target_names=TYPES, zero_division=0))
    print("confusion (rows=true, cols=pred):", TYPES, cm)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT)); tok.save_pretrained(str(OUT))
    res = {"val_macro_f1": best, "test_report": report, "test_confusion": cm, "labels": TYPES, "max_length": MAXLEN,
           "n": {"train": len(tr_x), "val": len(va_x), "test": len(te_x)}}
    (OUT / "type_meta.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    Path("eval/results/type_classifier_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
