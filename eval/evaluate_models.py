"""
Comprehensive Evaluation Pipeline for PromptShield X (Phase 3).
Evaluates:
  1. Rules-Only
  2. Old Fine-Tuned Model (distilbert_old)
  3. Retrained Model (distilbert)
  4. Combined Pipeline (Retrained Model + Rules + Normalizer)
Computes:
  - AUROC, AUPRC, Precision, Recall, F1, FPR, FNR
  - 1,000 bootstrap iterations for 95% Confidence Intervals
  - Hard-negative breakdown (Footnotes, Legal, Policy, Imperatives, Meta-discussions)
  - Frozen holdout benchmark comparison
Outputs structured evaluation results to eval/eval_results.json.
"""

import json
import re
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.modules.pattern_scanner import scan_prompt
from app.modules.normalizer import evaluate_normalized_evidence

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("eval/data")
WEIGHTS_DIR = Path("app/modules/weights")
OLD_MODEL_DIR = WEIGHTS_DIR / "distilbert_old"
NEW_MODEL_DIR = WEIGHTS_DIR / "distilbert"

def load_data(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def run_model_inference(model, tokenizer, texts: list[str], max_length: int = 128) -> np.ndarray:
    model.eval()
    probs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(DEVICE)
            outputs = model(**inputs)
            p = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p)
    return np.array(probs)

def run_rules_inference(texts: list[str]) -> np.ndarray:
    severities = []
    for t in texts:
        norm_ev = evaluate_normalized_evidence(t)
        severities.append(norm_ev["max_severity"] / 100.0)
    return np.array(severities)

def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_scores >= threshold).astype(int)
    
    # AUROC
    if len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_scores))
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        auprc = float(auc(recall_curve, precision_curve))
    else:
        auroc = 0.5
        auprc = 0.0

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
    }

def bootstrap_ci(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_list = {
        "auroc": [],
        "auprc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "fpr": [],
        "fnr": [],
    }

    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        sample_true = y_true[indices]
        sample_scores = y_scores[indices]
        
        # Guard if sample has only one class
        if len(np.unique(sample_true)) < 2:
            continue
            
        m = compute_metrics(sample_true, sample_scores, threshold=threshold)
        for k in metrics_list:
            metrics_list[k].append(m[k])

    ci_results = {}
    for k, vals in metrics_list.items():
        if len(vals) > 0:
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
            mean = float(np.mean(vals))
            ci_results[k] = {"mean": round(mean, 4), "ci_95": [round(low, 4), round(high, 4)]}
        else:
            ci_results[k] = {"mean": 0.0, "ci_95": [0.0, 0.0]}

    return ci_results

def evaluate_all():
    test_path = DATA_DIR / "test.json"
    print(f"Loading held-out test set from {test_path}...")
    test_data = load_data(test_path)

    texts = [d["text"] for d in test_data]
    y_true = np.array([d["label"] for d in test_data])
    is_hard_neg = np.array([d.get("is_hard_negative", False) for d in test_data])
    sources = np.array([d.get("source", "") for d in test_data])

    print(f"Held-out Test: {len(y_true)} total (Injections: {sum(y_true==1)}, Benign: {sum(y_true==0)}, Hard Negs: {sum(is_hard_neg)})")

    # 1. Rules-Only
    print("\n--- Evaluating Rules-Only ---")
    rules_scores = run_rules_inference(texts)
    rules_metrics = compute_metrics(y_true, rules_scores, threshold=0.5)
    rules_ci = bootstrap_ci(y_true, rules_scores, threshold=0.5)

    # 2. Old Model
    print("\n--- Evaluating Old Fine-Tuned Model ---")
    old_tok = AutoTokenizer.from_pretrained(str(OLD_MODEL_DIR))
    old_model = AutoModelForSequenceClassification.from_pretrained(str(OLD_MODEL_DIR)).to(DEVICE)
    old_scores = run_model_inference(old_model, old_tok, texts)
    old_metrics = compute_metrics(y_true, old_scores, threshold=0.5)
    old_ci = bootstrap_ci(y_true, old_scores, threshold=0.5)

    # 3. Retrained Model
    print("\n--- Evaluating Retrained Model ---")
    new_tok = AutoTokenizer.from_pretrained(str(NEW_MODEL_DIR))
    new_model = AutoModelForSequenceClassification.from_pretrained(str(NEW_MODEL_DIR)).to(DEVICE)
    new_scores = run_model_inference(new_model, new_tok, texts)
    new_metrics = compute_metrics(y_true, new_scores, threshold=0.5)
    new_ci = bootstrap_ci(y_true, new_scores, threshold=0.5)

    # 4. Combined Mode (Max of Retrained Model & Rules)
    print("\n--- Evaluating Combined Mode ---")
    combined_scores = np.maximum(new_scores, rules_scores)
    combined_metrics = compute_metrics(y_true, combined_scores, threshold=0.5)
    combined_ci = bootstrap_ci(y_true, combined_scores, threshold=0.5)

    # 5. Hard Negatives Breakdown
    hard_neg_indices = np.where(is_hard_neg)[0]
    hard_neg_y_true = y_true[hard_neg_indices]
    
    print("\n--- Hard Negatives Breakdown (150 samples) ---")
    hard_neg_results = {
        "rules_fpr": float(np.mean((rules_scores[hard_neg_indices] >= 0.5).astype(int))),
        "old_model_fpr": float(np.mean((old_scores[hard_neg_indices] >= 0.5).astype(int))),
        "new_model_fpr": float(np.mean((new_scores[hard_neg_indices] >= 0.5).astype(int))),
        "combined_fpr": float(np.mean((combined_scores[hard_neg_indices] >= 0.5).astype(int))),
    }
    print(f"Old Model Hard Negative FPR:   {hard_neg_results['old_model_fpr']*100:.1f}%")
    print(f"Retrained Model Hard Neg FPR: {hard_neg_results['new_model_fpr']*100:.1f}%")
    print(f"Rules-Only Hard Neg FPR:      {hard_neg_results['rules_fpr']*100:.1f}%")
    print(f"Combined Hard Neg FPR:        {hard_neg_results['combined_fpr']*100:.1f}%")

    # 6. Frozen Holdout Evaluation
    frozen_path = Path("eval/frozen_holdout.json")
    frozen_results = {}
    if frozen_path.exists():
        print("\n--- Evaluating Frozen Holdout Dataset (200 items) ---")
        with open(frozen_path, "r", encoding="utf-8") as f:
            frozen_data = json.load(f)
        
        fz_texts = [d["text"] for d in frozen_data]
        fz_y_true = np.array([1 if d["label"] in [1, "ATTACK", "INJECTION"] else 0 for d in frozen_data])
        
        fz_rules_scores = run_rules_inference(fz_texts)
        fz_old_scores = run_model_inference(old_model, old_tok, fz_texts)
        fz_new_scores = run_model_inference(new_model, new_tok, fz_texts)
        fz_combined_scores = np.maximum(fz_new_scores, fz_rules_scores)

        frozen_results = {
            "rules_only": compute_metrics(fz_y_true, fz_rules_scores),
            "old_model": compute_metrics(fz_y_true, fz_old_scores),
            "retrained_model": compute_metrics(fz_y_true, fz_new_scores),
            "combined": compute_metrics(fz_y_true, fz_combined_scores),
        }

    # Save summary
    eval_output = {
        "heldout_test_ci": {
            "rules_only": {"point": rules_metrics, "ci_95": rules_ci},
            "old_model": {"point": old_metrics, "ci_95": old_ci},
            "retrained_model": {"point": new_metrics, "ci_95": new_ci},
            "combined": {"point": combined_metrics, "ci_95": combined_ci},
        },
        "hard_negatives": hard_neg_results,
        "frozen_holdout": frozen_results,
    }

    out_file = Path("eval/evaluation_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)

    print(f"\nSaved complete evaluation report to {out_file}")

if __name__ == "__main__":
    evaluate_all()
