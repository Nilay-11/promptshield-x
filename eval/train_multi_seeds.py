"""
PromptShield X - Phase 3.5 Multi-Seed Training Pipeline (Seeds: 42, 123, 456).
Trains DistilBERT on train_v35.json, validates on val_v35.json.
Reports mean +/- std across seeds.
Saves the best overall seed model to app/modules/weights/distilbert/ and updates manifest.json.
"""

import json
import random
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

DATA_DIR = Path("eval/data")
WEIGHTS_DIR = Path("app/modules/weights")
MODEL_OUTPUT_DIR = WEIGHTS_DIR / "distilbert"
MANIFEST_PATH = WEIGHTS_DIR / "manifest.json"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 4e-5
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]


class PromptDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    avg_loss = total_loss / len(all_labels)
    auroc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "loss": avg_loss,
        "auroc": auroc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "probs": all_probs,
        "labels": all_labels,
    }


def find_threshold_at_target_fpr(labels: np.ndarray, probs: np.ndarray, target_fpr: float) -> float:
    """Finds lowest threshold where FPR <= target_fpr on the given dataset."""
    neg_probs = probs[labels == 0]
    if len(neg_probs) == 0:
        return 0.5
    sorted_neg = np.sort(neg_probs)
    idx = int(np.floor((1.0 - target_fpr) * len(sorted_neg)))
    idx = min(idx, len(sorted_neg) - 1)
    return float(sorted_neg[idx]) + 1e-4


def train_single_seed(seed: int, train_data: list[dict], val_data: list[dict], tokenizer):
    print(f"\n--- Training Seed {seed} ---")
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    train_dataset = PromptDataset(train_data, tokenizer, max_length=MAX_LENGTH)
    val_dataset = PromptDataset(val_data, tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        val_metrics = evaluate(model, val_loader, DEVICE)
        print(f"Seed {seed} | Epoch {epoch}/{EPOCHS} -> Val Loss: {val_metrics['loss']:.4f} | "
              f"AUROC: {val_metrics['auroc']:.4f} | F1: {val_metrics['f1']:.4f} | FPR: {val_metrics['fpr']*100:.2f}%")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    model.load_state_dict(best_state_dict)
    final_val = evaluate(model, val_loader, DEVICE)
    tau_1 = find_threshold_at_target_fpr(final_val["labels"], final_val["probs"], 0.01)
    tau_5 = find_threshold_at_target_fpr(final_val["labels"], final_val["probs"], 0.05)

    final_val["tau_fpr1"] = tau_1
    final_val["tau_fpr5"] = tau_5

    return model, final_val


def run_multi_seed_pipeline():
    with open(DATA_DIR / "train_v35.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "val_v35.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"Loaded train_v35: {len(train_data)} | val_v35: {len(val_data)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = {}
    best_seed = None
    best_loss = float("inf")
    best_model = None

    for seed in SEEDS:
        model, val_m = train_single_seed(seed, train_data, val_data, tokenizer)
        results[seed] = {
            "loss": float(val_m["loss"]),
            "auroc": float(val_m["auroc"]),
            "f1": float(val_m["f1"]),
            "precision": float(val_m["precision"]),
            "recall": float(val_m["recall"]),
            "fpr": float(val_m["fpr"]),
            "tau_fpr1": float(val_m["tau_fpr1"]),
            "tau_fpr5": float(val_m["tau_fpr5"]),
        }
        if val_m["loss"] < best_loss:
            best_loss = val_m["loss"]
            best_seed = seed
            best_model = model

    # Compute mean and standard deviation
    summary = {}
    for metric in ["loss", "auroc", "f1", "precision", "recall", "fpr", "tau_fpr1", "tau_fpr5"]:
        vals = [results[s][metric] for s in SEEDS]
        summary[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": vals,
        }

    print("\n=======================================================")
    print("3-SEED VALIDATION SUMMARY (Seeds: 42, 123, 456)")
    print("=======================================================")
    for m, s in summary.items():
        print(f"  {m:12s}: {s['mean']:.4f} +/- {s['std']:.4f}")
    print(f"  Best seed: {best_seed} (val loss: {best_loss:.4f})")

    # Save best model to app/modules/weights/distilbert
    print(f"\nSaving best model (Seed {best_seed}) to {MODEL_OUTPUT_DIR}...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model.save_pretrained(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    # Compute SHA-256
    hasher = hashlib.sha256()
    for p in sorted(MODEL_OUTPUT_DIR.glob("**/*")):
        if p.is_file():
            hasher.update(p.read_bytes())
    new_hash = hasher.hexdigest()

    manifest_data = {
        "distilbert": {
            "version": "3.5.0-multiseed-retrained",
            "sha256": new_hash,
            "best_seed": best_seed,
            "architecture": "distilbert-base-uncased",
            "multi_seed_summary": summary,
            "validation_metrics": results[best_seed],
        }
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    with open("eval/multi_seed_results.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": results, "summary": summary, "best_seed": best_seed}, f, indent=2)

    print(f"[Manifest] Updated {MANIFEST_PATH} with SHA256: {new_hash}")
    print("Multi-seed training complete!")


if __name__ == "__main__":
    run_multi_seed_pipeline()
