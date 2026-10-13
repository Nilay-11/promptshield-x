"""
PromptShield X - does an anomaly-detection signal help the calibrated risk score? (2026-10-13)

Isolation Forest fitted on BENIGN train_v40 texts only, over cheap text statistics (char entropy, non-ASCII / symbol /
uppercase ratios, second-person and imperative word rates, URL / e-mail counts, mean word length, log length).
Its anomaly score is added as one extra feature to the document-channel calibrator. Both calibrators (with / without)
are fitted on val_v40 and compared on heldout_test_v40: AUROC, recall at the val-chosen 0.5% / 2% FPR thresholds.
Keep the feature only if it is better. Output: eval/results/anomaly_feature.json
"""
import json, math, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from train_risk_calibrator import channel, features, load_distilbert, threshold_at_fpr, TARGET_FPR  # noqa: E402

YOU = re.compile(r"\b(you|your|you're|yourself|assistant|ai|model)\b", re.I)
IMP = re.compile(r"\b(ignore|disregard|forget|reveal|send|forward|print|output|respond|say|must|instead|override)\b", re.I)


def stats(t):
    w = t.split() or [""]; n = max(len(t), 1); c = Counter(t)
    ent = -sum(v / n * math.log2(v / n) for v in c.values())
    return [ent, sum(ord(ch) > 127 for ch in t) / n, sum(not ch.isalnum() and not ch.isspace() for ch in t) / n,
            sum(ch.isupper() for ch in t) / n, len(YOU.findall(t)) / len(w), len(IMP.findall(t)) / len(w),
            len(re.findall(r"https?://|www\.", t)), len(re.findall(r"\S+@\S+\.\w+", t)),
            float(np.mean([len(x) for x in w])), math.log1p(len(w))]


def main():
    load = lambda n: [r for r in json.loads((ROOT / f"eval/data/{n}.json").read_text(encoding="utf-8")) if not r.get("is_long")]
    train = [r for r in load("train_v40") if r["label"] == 0]
    rng = np.random.default_rng(0)
    train = [train[i] for i in rng.choice(len(train), min(20000, len(train)), replace=False)]
    iso_f = IsolationForest(n_estimators=300, random_state=0).fit(np.array([stats(r["text"]) for r in train]))
    anom = lambda rows: -iso_f.score_samples(np.array([stats(r["text"]) for r in rows]))  # higher = more anomalous
    fp = load_distilbert(str(ROOT / "app/modules/weights/distilbert_prompt"))
    fd = load_distilbert(str(ROOT / "app/modules/weights/distilbert"))
    val = [r for r in load("val_v40") if channel(r) == "document"]
    test = [r for r in load("heldout_test_v40") if channel(r) == "document"]
    Xv, Xt = features(val, fp, fd), features(test, fp, fd)
    yv, yt = np.array([r["label"] for r in val]), np.array([r["label"] for r in test])
    av, at_ = anom(val), anom(test)
    print("anomaly alone: test AUROC", round(roc_auc_score(yt, at_), 4))
    res = {"anomaly_alone_test_auroc": round(float(roc_auc_score(yt, at_)), 4)}
    for name, (A, B) in {"baseline": (Xv, Xt), "with_anomaly": (np.c_[Xv, av], np.c_[Xt, at_])}.items():
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(A, yv)
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(clf.decision_function(A), yv)
        pv, pt = iso.predict(clf.decision_function(A)), iso.predict(clf.decision_function(B))
        r = {"test_auroc_logistic": round(float(roc_auc_score(yt, clf.decision_function(B))), 4)}
        for k, f in TARGET_FPR.items():
            tau = threshold_at_fpr(yv, pv, f)
            r[k] = {"recall": round(float((pt[yt == 1] >= tau).mean()), 4), "fpr": round(float((pt[yt == 0] >= tau).mean()), 4)}
        res[name] = r; print(name, r)
    (ROOT / "eval/results/anomaly_feature.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
