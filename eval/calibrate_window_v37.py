"""
PromptShield X - calibrate the reading-order window threshold on VALIDATION PDFs only.

The v37 val run showed the production pipeline at 2.8% FPR (target 1%), partly because the
reading-order window pass joins a hidden benign sentence with the body text and flags the join.
This scans format_suite_v36_val PDFs once, records each document's max per-segment risk and each
flagged window's semantic score, and picks the lowest window semantic threshold that keeps the
document-level val FPR <= 1% (falls back to the best achievable) while maximizing recall.
The chosen value is written to eval/results/window_calibration_v37.json; apply it via
settings.reading_order_window_min_semantic.
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
OUT = PROJECT_ROOT / "eval/results/window_calibration_v37.json"


def main():
    specs = [s for s in json.loads(DATA.read_text(encoding="utf-8")) if s["format"] == "pdf"]
    settings.reading_order_window_min_semantic = 0  # observe every window, then choose the threshold
    scanner = DocumentScanner()
    seg_max, win_max, labels = [], [], []
    for i, s in enumerate(specs):
        r = scanner.scan_pdf_bytes(render_pdf(s), filename="val.pdf")
        seg_max.append(max([x.get("final_risk", 0) for x in r["all_segments"]] or [0]))
        win_max.append(max([w.get("semantic_attack_score", 0) for w in r.get("window_triggers", [])] or [0]))
        labels.append(s["label"])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(specs)}", flush=True)
    seg_max, win_max, labels = map(np.array, (seg_max, win_max, labels))
    review = settings.review_risk_threshold

    def at(t):
        flag = (seg_max >= review) | (win_max >= t)
        return float(flag[labels == 1].mean()), float(flag[labels == 0].mean())

    grid = {t: at(t) for t in range(0, 102)}
    ok = [t for t, (_, f) in grid.items() if f <= 0.01]
    chosen = min(ok) if ok else min(grid, key=lambda t: (grid[t][1], -grid[t][0]))
    report = {
        "val_pdfs": len(specs),
        "segments_only": {"recall": float((seg_max >= review)[labels == 1].mean()),
                          "fpr": float((seg_max >= review)[labels == 0].mean())},
        "all_windows": {"recall": at(1)[0], "fpr": at(1)[1]},
        "chosen_window_min_semantic": chosen,
        "at_chosen": {"recall": grid[chosen][0], "fpr": grid[chosen][1]},
        "grid": {t: {"recall": round(r, 4), "fpr": round(f, 4)} for t, (r, f) in grid.items() if t % 5 == 0},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2))


if __name__ == "__main__":
    main()
