"""
PromptShield X - check / calibrate the production DocumentScanner thresholds on VALIDATION data only.

Renders format_suite_v36_val PDFs (val documents + val attacks; never heldout or final), runs the
production DocumentScanner (rules + model + structural amplifier, current production weights) and
reports recall / FPR of REVIEW-or-BLOCK and BLOCK at the configured thresholds, plus the
lowest review threshold whose val FPR <= 1% and <= 5%.

Usage:
    python eval/calibrate_production_v36.py
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_documents_v36 import render_pdf
from app.core.settings import settings
from app.modules.document_scanner import DocumentScanner

DATA = PROJECT_ROOT / "eval/data/format_suite_v36_val.json"
OUT = PROJECT_ROOT / "eval/results/production_calibration_v36.json"


def lowest_threshold(scores, labels, target):
    """Smallest integer threshold t such that FPR(score >= t) <= target."""
    neg = scores[labels == 0]
    for t in range(0, 101):
        if np.mean(neg >= t) <= target:
            return t
    return 101


def main():
    specs = [s for s in json.loads(DATA.read_text(encoding="utf-8")) if s["format"] == "pdf"]
    scanner = DocumentScanner()
    scores, labels = [], []
    for i, s in enumerate(specs):
        res = scanner.scan_pdf_bytes(render_pdf(s), filename="val.pdf")
        scores.append(res["overall_risk_score"])
        labels.append(s["label"])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(specs)}", flush=True)
    scores, labels = np.array(scores), np.array(labels)

    def at(t):
        flag = scores >= t
        return {"recall": round(float(flag[labels == 1].mean()), 4), "fpr": round(float(flag[labels == 0].mean()), 4)}

    report = {
        "val_pdfs": len(specs),
        "configured": {"review_risk_threshold": settings.review_risk_threshold,
                       "block_risk_threshold": settings.block_risk_threshold},
        "at_review_threshold": at(settings.review_risk_threshold),
        "at_block_threshold": at(settings.block_risk_threshold),
        "lowest_review_threshold_val_fpr1": lowest_threshold(scores, labels, 0.01),
        "lowest_review_threshold_val_fpr5": lowest_threshold(scores, labels, 0.05),
        "benign_score_quantiles_50_90_95_99": np.percentile(scores[labels == 0], [50, 90, 95, 99]).tolist(),
        "attack_score_quantiles_1_5_10_50": np.percentile(scores[labels == 1], [1, 5, 10, 50]).tolist(),
    }
    for t in (report["lowest_review_threshold_val_fpr1"], report["lowest_review_threshold_val_fpr5"]):
        report[f"at_{t}"] = at(t)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
