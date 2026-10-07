"""
PromptShield X - calibrated risk score.

Replaces the hand-set risk formula (category base severity x model confidence, blended with rule severity)
with a probability learned from data: risk = 100 x P(attack | signals).

Signals per text: logit of the prompt detector (v37), logit of the document detector (v40), max rule severity,
any-rule flag, log word count (the prompt channel does not use the document detector). One logistic calibrator per channel:
  prompt   - text a user types (deepset, Gandalf, jailbreak / goal-swapped standalone, Dolly, no_robots, awesome ...)
  document - text from files / RAG / tool output (doc chunks, BIPIA, LLMail, InjecAgent, RAG chunks ...)
Fitted on val_v40 only; measured on heldout_test_v40 (ECE, Brier, reliability bins). REVIEW / BLOCK thresholds are
picked on val as probabilities at a target false-positive rate. Output: app/modules/weights/risk_calibrator.json.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from evaluate_v36 import load_distilbert  # noqa: E402
from app.modules.pattern_scanner import scan_prompt  # noqa: E402

DATA = ROOT / "eval/data"
OUT = ROOT / "app/modules/weights/risk_calibrator.json"
RESULTS = ROOT / "eval/results/risk_calibration.json"
PROMPT_SOURCES = ("deepset", "gandalf", "rubend18", "jackhhao", "tensor_trust", "goalswap_standalone", "wasp_standalone",
                  "satml", "dolly", "no_robots", "awesome_prompts")
TARGET_FPR = {"review": 0.02, "block": 0.005}


def channel(r):
    return "prompt" if r["source"].startswith(PROMPT_SOURCES) and not r.get("doc_injected") else "document"


def logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def features(rows, fp, fd):
    texts = [r["text"] for r in rows]
    pp, pd = fp(texts), fd(texts)
    X = []
    for t, a, b in zip(texts, pp, pd):
        sev = scan_prompt(t).get("severity", 0)
        X.append([logit(a), logit(b), sev / 100.0, 1.0 if sev > 0 else 0.0, math.log1p(len(t.split()))])
    return np.array(X)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
            table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_predicted": round(float(p[m].mean()), 3),
                          "actual_attack_rate": round(float(y[m].mean()), 3)})
    return tot, table


def threshold_at_fpr(y, p, target):
    neg = np.sort(p[y == 0])
    return float(neg[min(int(np.floor((1 - target) * len(neg))), len(neg) - 1)]) + 1e-6


def main():
    fp = load_distilbert(str(ROOT / "app/modules/weights/distilbert_prompt"))
    fd = load_distilbert(str(ROOT / "app/modules/weights/distilbert"))
    val = [r for r in json.loads((DATA / "val_v40.json").read_text(encoding="utf-8")) if not r.get("is_long")]
    test = [r for r in json.loads((DATA / "heldout_test_v40.json").read_text(encoding="utf-8")) if not r.get("is_long")]
    out, report = {"features": ["logit_p_prompt", "logit_p_document", "rule_severity", "rule_hit", "log_words"]}, {}
    for ch in ("prompt", "document"):
        v = [r for r in val if channel(r) == ch]
        t = [r for r in test if channel(r) == ch]
        Xv, yv = features(v, fp, fd), np.array([r["label"] for r in v])
        Xt, yt = features(t, fp, fd), np.array([r["label"] for r in t])
        if ch == "prompt":
            # The document detector over-scores polite user requests ("Translate this email into French": 0.92),
            # which validation data under-represents. Prompts are judged by the prompt detector + rules only.
            Xv[:, 1] = 0.0
            Xt[:, 1] = 0.0
            # Length is a shortcut in validation prompts (most short ones are Gandalf attacks), so it is dropped:
            # with it, "Explain how photosynthesis works." scored 57%.
            Xv[:, 4] = 0.0
            Xt[:, 4] = 0.0
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xv, yv)
        # Isotonic step on top of the logistic score: a monotone curve fitted to validation outcomes. The detectors'
        # probabilities are bunched near 0 and 1, which a single logistic curve cannot map to true attack rates.
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(clf.decision_function(Xv), yv)
        pv, pt = iso.predict(clf.decision_function(Xv)), iso.predict(clf.decision_function(Xt))
        e, table = ece(yt, pt)
        thr = {k: threshold_at_fpr(yv, pv, f) for k, f in TARGET_FPR.items()}
        at = lambda tau: {"recall": round(float((pt[yt == 1] >= tau).mean()), 4), "fpr": round(float((pt[yt == 0] >= tau).mean()), 4)}
        # the old hand-set scale, for comparison: max model prob as a probability
        old_p = 1 / (1 + np.exp(-(Xt[:, 0] if ch == "prompt" else Xt[:, 1])))
        out[ch] = {"coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                   "fit_attack_rate": float(yv.mean()),  # base rate the probabilities are calibrated to (for prior correction)
                   "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
                   "review_threshold": thr["review"], "block_threshold": thr["block"]}
        report[ch] = {"n_val": len(v), "n_test": len(t), "test_auroc": round(float(roc_auc_score(yt, pt)), 4),
                      "test_ece": round(float(e), 4), "test_brier": round(float(brier_score_loss(yt, pt)), 4),
                      "uncalibrated_model_ece": round(float(ece(yt, old_p)[0]), 4),
                      "thresholds": {k: round(v_, 4) for k, v_ in thr.items()},
                      "test_at_review": at(thr["review"]), "test_at_block": at(thr["block"]), "reliability": table}
        print(f"\n[{ch}] val {len(v)} test {len(t)} | AUROC {report[ch]['test_auroc']} | ECE {report[ch]['test_ece']} "
              f"(raw model {report[ch]['uncalibrated_model_ece']}) | Brier {report[ch]['test_brier']}")
        print(f"   thresholds review {thr['review']:.3f} block {thr['block']:.3f} | test review {at(thr['review'])} block {at(thr['block'])}")
        for row in table:
            print(f"   {row['bin']}  n={row['n']:5d}  predicted {row['mean_predicted']:.3f}  actual {row['actual_attack_rate']:.3f}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT} and {RESULTS}")


if __name__ == "__main__":
    main()
