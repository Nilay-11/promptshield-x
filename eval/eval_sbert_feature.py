"""
PromptShield X - does a Sentence-BERT signal help the calibrated risk score? (2026-10-15)

all-MiniLM-L6-v2 sentence embeddings; a logistic-regression head is trained on a 30k sample of train_v40 (both classes).
Its attack probability (logit) is added as one extra feature to the document-channel calibrator. Both calibrators
(with / without) are fitted on val_v40 and compared on heldout_test_v40, exactly like eval_anomaly_feature.py.
Keep the feature only if it is better. Output: eval/results/sbert_feature.json
"""
import json, math, sys
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from train_risk_calibrator import channel, features, load_distilbert, threshold_at_fpr, TARGET_FPR  # noqa: E402


def main():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    emb = lambda rows: st.encode([r["text"] for r in rows], batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    load = lambda n: [r for r in json.loads((ROOT / f"eval/data/{n}.json").read_text(encoding="utf-8")) if not r.get("is_long")]
    train = load("train_v40"); rng = np.random.default_rng(0)
    train = [train[i] for i in rng.choice(len(train), min(30000, len(train)), replace=False)]
    head = LogisticRegression(C=1.0, max_iter=3000).fit(emb(train), [r["label"] for r in train])
    sb = lambda rows: head.decision_function(emb(rows))
    fp = load_distilbert(str(ROOT / "app/modules/weights/distilbert_prompt"))
    fd = load_distilbert(str(ROOT / "app/modules/weights/distilbert"))
    val = [r for r in load("val_v40") if channel(r) == "document"]
    test = [r for r in load("heldout_test_v40") if channel(r) == "document"]
    Xv, Xt = features(val, fp, fd), features(test, fp, fd)
    yv, yt = np.array([r["label"] for r in val]), np.array([r["label"] for r in test])
    sv, stt = sb(val), sb(test)
    res = {"sbert_alone_test_auroc": round(float(roc_auc_score(yt, stt)), 4)}
    print("sbert alone: test AUROC", res["sbert_alone_test_auroc"])
    for name, (A, B) in {"baseline": (Xv, Xt), "with_sbert": (np.c_[Xv, sv], np.c_[Xt, stt])}.items():
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(A, yv)
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(clf.decision_function(A), yv)
        pv, pt = iso.predict(clf.decision_function(A)), iso.predict(clf.decision_function(B))
        r = {"test_auroc_logistic": round(float(roc_auc_score(yt, clf.decision_function(B))), 4)}
        for k, f in TARGET_FPR.items():
            tau = threshold_at_fpr(yv, pv, f)
            r[k] = {"recall": round(float((pt[yt == 1] >= tau).mean()), 4), "fpr": round(float((pt[yt == 0] >= tau).mean()), 4)}
        res[name] = r; print(name, r)
    (ROOT / "eval/results/sbert_feature.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("RESULT done")


if __name__ == "__main__":
    main()
