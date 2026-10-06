"""
PromptShield X - Phase 3.5 Comprehensive Benchmark on Held-Out Test Set (eval/data/heldout_test_v35.json).
Evaluates:
1. Retrained DistilBERT (Phase 3.5, Seed 123)
2. Rules-Only System
3. Combined System (Model + Rules)
4. ProtectAI DeBERTa Baseline: protectai/deberta-v3-base-prompt-injection-v2
5. Deepset DeBERTa Baseline: deepset/deberta-v3-base-injection
Evaluates metrics:
- ROC-AUC
- Expected Calibration Error (ECE)
- Operating points:
  * Default threshold (tau = 0.50)
  * Val-calibrated FPR <= 1% threshold (tau_fpr1 = 0.4047)
  * Val-calibrated FPR <= 5% threshold (tau_fpr5 = 0.0213)
  * Matched-FPR operating points: Recall at 1% FPR and Recall at 5% FPR
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from app.modules.pattern_scanner import scan_prompt
from app.modules.distilbert_classifier import classify_prompt_distilbert
from app.core.settings import settings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PATH = Path("eval/data/heldout_test_v35.json")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / total) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_metrics_at_threshold(labels: np.ndarray, probs: np.ndarray, tau: float) -> dict:
    preds = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tau": float(tau),
        "recall": float(rec),
        "precision": float(prec),
        "fpr": float(fpr),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def find_recall_at_matched_fpr(labels: np.ndarray, probs: np.ndarray, target_fpr: float) -> dict:
    """Finds recall at the empirical threshold that achieves target_fpr on this test set."""
    neg_probs = probs[labels == 0]
    sorted_neg = np.sort(neg_probs)
    idx = int(np.floor((1.0 - target_fpr) * len(sorted_neg)))
    idx = min(idx, len(sorted_neg) - 1)
    emp_tau = float(sorted_neg[idx]) + 1e-4
    m = compute_metrics_at_threshold(labels, probs, emp_tau)
    m["target_fpr"] = target_fpr
    return m


def evaluate_transformer_model(model_id_or_path: str, samples: list[dict], is_hf_hub: bool = False):
    print(f"\nEvaluating Transformer: {model_id_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    model.to(DEVICE)
    model.eval()

    probs = []
    labels = []

    with torch.no_grad():
        for s in samples:
            text = s["text"]
            lbl = s["label"]
            inputs = tokenizer(text, truncation=True, max_length=128, padding=False, return_tensors="pt").to(DEVICE)
            out = model(**inputs)
            logits = out.logits
            # ProtectAI uses label 1 as injection / label 0 as safe
            p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            # If model config defines id2label or 2 classes
            prob_inj = float(p[1]) if len(p) > 1 else float(p[0])
            probs.append(prob_inj)
            labels.append(lbl)

    labels = np.array(labels)
    probs = np.array(probs)
    auroc = float(roc_auc_score(labels, probs))
    ece = compute_ece(probs, labels)

    return {
        "auroc": auroc,
        "ece": ece,
        "probs": probs,
        "labels": labels,
    }


def run_benchmarks():
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    labels = np.array([s["label"] for s in samples])
    print(f"Loaded heldout_test_v35: {len(samples)} samples (Inj: {np.sum(labels==1)}, Benign: {np.sum(labels==0)})")

    # 1. DistilBERT Retrained (Phase 3.5 Seed 123)
    distilbert_res = evaluate_transformer_model("app/modules/weights/distilbert", samples)

    # 2. ProtectAI DeBERTa Baseline
    protectai_res = evaluate_transformer_model("protectai/deberta-v3-base-prompt-injection-v2", samples)

    # 3. Deepset DeBERTa Baseline
    deepset_res = evaluate_transformer_model("deepset/deberta-v3-base-injection", samples)

    # 4. Rules-Only System
    print("\nEvaluating Rules-Only System...")
    rules_probs = []
    for s in samples:
        scan = scan_prompt(s["text"])
        rules_probs.append(scan["severity"] / 100.0)
    rules_probs = np.array(rules_probs)
    rules_auroc = float(roc_auc_score(labels, rules_probs))
    rules_ece = compute_ece(rules_probs, labels)
    rules_res = {"auroc": rules_auroc, "ece": rules_ece, "probs": rules_probs, "labels": labels}

    # 5. Combined System (DistilBERT + Normalizer + Rules)
    print("\nEvaluating Combined System...")
    settings.classifier_mode = "combined"
    combined_probs = []
    for s in samples:
        c_res = classify_prompt_distilbert(s["text"])
        combined_probs.append(c_res["final_score"] / 100.0)
    combined_probs = np.array(combined_probs)
    combined_auroc = float(roc_auc_score(labels, combined_probs))
    combined_ece = compute_ece(combined_probs, labels)
    combined_res = {"auroc": combined_auroc, "ece": combined_ece, "probs": combined_probs, "labels": labels}

    # Assemble Benchmark Results Table
    systems = {
        "PromptShield-X Retrained (DistilBERT)": distilbert_res,
        "Rules-Only": rules_res,
        "PromptShield-X Combined (Hybrid)": combined_res,
        "ProtectAI deberta-v3-base": protectai_res,
        "Deepset deberta-v3-base": deepset_res,
    }

    report = {}

    print("\n" + "=" * 90)
    print("PHASE 3.5 COMPREHENSIVE BENCHMARK TABLE (HELD-OUT TEST SET)")
    print("=" * 90)
    print(f"{'System / Model':36s} | {'AUROC':7s} | {'ECE':7s} | {'Rec@1%FPR':9s} | {'Rec@5%FPR':9s} | {'FPR@Default':11s} | {'Rec@Default':11s}")
    print("-" * 90)

    for name, res in systems.items():
        p = res["probs"]
        y = res["labels"]

        m_default = compute_metrics_at_threshold(y, p, 0.50)
        m_fpr1 = find_recall_at_matched_fpr(y, p, 0.01)
        m_fpr5 = find_recall_at_matched_fpr(y, p, 0.05)

        report[name] = {
            "auroc": res["auroc"],
            "ece": res["ece"],
            "default_0.50": m_default,
            "matched_fpr1": m_fpr1,
            "matched_fpr5": m_fpr5,
        }

        print(f"{name:36s} | {res['auroc']:7.4f} | {res['ece']:7.4f} | "
              f"{m_fpr1['recall']*100:8.2f}% | {m_fpr5['recall']*100:8.2f}% | "
              f"{m_default['fpr']*100:10.2f}% | {m_default['recall']*100:10.2f}%")

    with open("eval/benchmark_phase3_5_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nResults successfully saved to eval/benchmark_phase3_5_results.json")


if __name__ == "__main__":
    run_benchmarks()
