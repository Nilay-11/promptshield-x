"""
DistilBERT Fine-Tuning Pipeline for PromptShield X.
Trains distilbert-base-uncased on source-split train.json with hard negatives.
Evaluates on val.json to select the best checkpoint.
Saves retrained weights to app/modules/weights/distilbert/ and updates manifest.json.
"""

import json
import os
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

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_DIR = Path("eval/data")
WEIGHTS_DIR = Path("app/modules/weights")
MODEL_OUTPUT_DIR = WEIGHTS_DIR / "distilbert"
MANIFEST_PATH = WEIGHTS_DIR / "manifest.json"

MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray, target_fpr: float = 0.01) -> float:
    """Finds lowest decision threshold where validation FPR <= target_fpr."""
    neg_probs = probs[labels == 0]
    if len(neg_probs) == 0:
        return 0.5

    sorted_neg = np.sort(neg_probs)
    idx = int(np.floor((1.0 - target_fpr) * len(sorted_neg)))
    idx = min(idx, len(sorted_neg) - 1)
    threshold = float(sorted_neg[idx]) + 1e-4
    return min(max(threshold, 0.5), 0.99)


def train():
    print(f"=== DistilBERT Retraining Pipeline ===")
    print(f"Device: {DEVICE}")

    train_path = DATA_DIR / "train.json"
    val_path = DATA_DIR / "val.json"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
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

        avg_train_loss = total_train_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, DEVICE)

        print(f"Epoch {epoch}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val AUROC: {val_metrics['auroc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f} | Val FPR: {val_metrics['fpr']*100:.2f}%")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            print(f"  [*] New best validation loss! Saving checkpoint to {MODEL_OUTPUT_DIR}...")
            model.save_pretrained(str(MODEL_OUTPUT_DIR))
            tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    # Evaluate final best model on val set to determine optimal threshold
    best_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_OUTPUT_DIR))
    best_model.to(DEVICE)
    final_val_metrics = evaluate(best_model, val_loader, DEVICE)

    optimal_tau = find_optimal_threshold(final_val_metrics["labels"], final_val_metrics["probs"], target_fpr=0.01)
    print(f"\n[Validation Tuning] Selected threshold for FPR <= 1.0%: tau = {optimal_tau:.4f}")

    # Compute SHA-256 for manifest
    hasher = hashlib.sha256()
    for p in sorted(MODEL_OUTPUT_DIR.glob("**/*")):
        if p.is_file():
            hasher.update(p.read_bytes())
    new_hash = hasher.hexdigest()

    # Update manifest.json
    manifest_data = {
        "distilbert": {
            "version": "2.0.0-phase3-retrained",
            "sha256": new_hash,
            "architecture": "distilbert-base-uncased",
            "validation_metrics": {
                "val_loss": round(final_val_metrics["loss"], 4),
                "val_auroc": round(final_val_metrics["auroc"], 4),
                "val_f1": round(final_val_metrics["f1"], 4),
                "val_fpr": round(final_val_metrics["fpr"], 4),
                "optimal_threshold": round(optimal_tau, 4)
            }
        }
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[Manifest] Updated {MANIFEST_PATH} with SHA256: {new_hash}")
    print("=== Training Complete ===")


if __name__ == "__main__":
    train()
