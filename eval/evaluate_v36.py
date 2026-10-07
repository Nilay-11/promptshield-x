"""
PromptShield X - Phase 3.6 evaluation on the cleaned v36 datasets.

For every scorer (style-feature LR, TF-IDF LR, and each DistilBERT checkpoint):
  * thresholds are chosen on val_v36 only (0.5, FPR<=1%, FPR<=5%) and applied unchanged.
    Short inputs and long documents (sliding windows, max score) get separate thresholds, each
    chosen on the matching part of val
  * per-source recall / FPR with 95% cluster-bootstrap CIs over `group` (unique documents /
    strings, not rows)
  * slices: source-balanced sources, persona benign, context-dependent, long documents
    (sliding windows, max score), v35 template probes, ambiguous rows
  * per-sample scores are written to eval/results/v36_eval_samples.jsonl

The baselines are always reported next to the model.

Usage:
    python eval/evaluate_v36.py                       # heldout_test_v36 (dev test)
    python eval/evaluate_v36.py --final               # final_test_v36, allowed ONCE
    python eval/evaluate_v36.py --models app/modules/weights/distilbert models/distilbert_v36
    python eval/evaluate_v36.py --data v37            # v37 splits (models/distilbert_v37 by default)
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("eval/data")
RESULTS_DIR = Path("eval/results")
DEFAULT_MODELS = ["app/modules/weights/distilbert", "models/distilbert_{data}"]

WINDOW_CHARS, STRIDE_CHARS = 800, 400
N_BOOT = 1000
BALANCED_PREFIXES = ("deepset", "bipia", "ledgar", "unfair_tos", "enron_email", "wikipedia", "arxiv",
                     "long_", "padded_", "jackhhao", "llmail", "bbc_news")


def load(name):
    p = DATA_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def windows(text):
    if len(text) <= WINDOW_CHARS:
        return [text]
    return [text[s : s + WINDOW_CHARS] for s in range(0, len(text) - STRIDE_CHARS, STRIDE_CHARS)]


def score_rows(score_texts, rows):
    """Score short rows directly and long rows as max over sliding windows."""
    flat, owner = [], []
    for i, r in enumerate(rows):
        parts = windows(r["text"]) if r.get("is_long") else [r["text"]]
        flat += parts
        owner += [i] * len(parts)
    s = score_texts(flat)
    out = np.full(len(rows), -np.inf)
    for i, v in zip(owner, s):
        out[i] = max(out[i], v)
    return out


# ----------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------
def style_features(texts):
    f = []
    for t in texts:
        n = max(len(t), 1)
        words = t.split()
        f.append([
            np.log1p(len(t)),
            np.log1p(len(words)),
            sum(c.isupper() for c in t) / n,
            sum(not c.isalnum() and not c.isspace() for c in t) / n,
            t.count("\n") / n * 100,
        ])
    return np.array(f)


def fit_style(train):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(style_features([r["text"] for r in train]), [r["label"] for r in train])
    return lambda texts: clf.predict_proba(style_features(texts))[:, 1]


def fit_tfidf(train):
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100000),
                        LogisticRegression(max_iter=2000, C=4.0))
    clf.fit([r["text"] for r in train], [r["label"] for r in train])
    return lambda texts: clf.predict_proba(texts)[:, 1]


def load_distilbert(path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).eval().to(device)
    meta_path = Path(path) / "training_meta.json"
    max_len = json.loads(meta_path.read_text())["max_length"] if meta_path.exists() else 128

    @torch.no_grad()
    def score(texts):
        """Token-level sliding windows (stride = half the window), max over windows,
        so nothing past max_length is silently truncated away."""
        enc = tok(texts, truncation=True, max_length=max_len, stride=max_len // 2,
                  return_overflowing_tokens=True)
        owner = enc["overflow_to_sample_mapping"]
        windows_ = enc["input_ids"]
        order = sorted(range(len(windows_)), key=lambda i: len(windows_[i]))
        probs = np.zeros(len(windows_))
        for s in range(0, len(order), 64):
            ids = order[s : s + 64]
            batch = tok.pad({"input_ids": [windows_[i] for i in ids]}, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(**batch).logits
            probs[ids] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        out = np.zeros(len(texts))
        for w, i in enumerate(owner):
            out[i] = max(out[i], probs[w])
        return out

    return score


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def threshold_at_fpr(labels, probs, target):
    neg = np.sort(probs[labels == 0])
    k = min(int(np.floor((1.0 - target) * len(neg))), len(neg) - 1)
    return float(neg[k]) + 1e-6


def rate(mask_hits, mask_den):
    d = mask_den.sum()
    return float(mask_hits[mask_den].mean()) if d else None


def cluster_boot(labels, probs, groups, tau, rng):
    """Recall and FPR with 95% CIs from resampling groups. `tau` may be per-row."""
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    pred = probs >= tau
    point = {"recall": rate(pred, labels == 1), "fpr": rate(pred, labels == 0)}
    boots = defaultdict(list)
    for _ in range(N_BOOT):
        sample = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq))])
        for k, v in (("recall", rate(pred[sample], labels[sample] == 1)),
                     ("fpr", rate(pred[sample], labels[sample] == 0))):
            if v is not None:
                boots[k].append(v)
    out = {}
    for k, v in point.items():
        if v is None:
            continue
        lo, hi = (np.percentile(boots[k], [2.5, 97.5]) if boots[k] else (v, v))
        out[k] = {"value": round(v, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def auroc(labels, probs):
    return round(float(roc_auc_score(labels, probs)), 4) if len(set(labels)) == 2 else None


def evaluate_slice(rows, probs, taus, rng):
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    is_long = np.array([bool(r.get("is_long")) for r in rows])
    res = {"n": len(rows), "groups": int(len(np.unique(groups))),
           "n_injection": int(labels.sum()), "n_benign": int((1 - labels).sum()),
           "auroc": auroc(labels, probs)}
    for name, t in taus.items():
        tau = np.where(is_long, t["long"], t["short"])
        res[f"at_{name}"] = cluster_boot(labels, probs, groups, tau, rng)
    return res


def choose_thresholds(val, vp):
    labels = np.array([r["label"] for r in val])
    is_long = np.array([bool(r.get("is_long")) for r in val])
    taus = {"tau_0.5": {"short": 0.5, "long": 0.5}}
    for name, target in (("val_fpr1", 0.01), ("val_fpr5", 0.05)):
        short = threshold_at_fpr(labels[~is_long], vp[~is_long], target)
        long_ = threshold_at_fpr(labels[is_long], vp[is_long], target) if (is_long & (labels == 0)).any() else short
        taus[name] = {"short": short, "long": long_}
    return taus


def slices(rows):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by["ALL"].append(i)
        by[f"source:{r['source']}"].append(i)
        if r["source"].startswith(BALANCED_PREFIXES):
            by["slice:source_balanced"].append(i)
        if r.get("persona") and r["label"] == 0:
            by["slice:persona_benign"].append(i)
        if r.get("context_dependent"):
            by["slice:context_dependent"].append(i)
        if r.get("doc_injected"):
            by["slice:doc_injected"].append(i)
        if r.get("meta_discussion"):
            by["slice:meta_discussion"].append(i)
        if r.get("is_long"):
            by["slice:long_documents"].append(i)
        if not r.get("is_long"):
            by["slice:short_only"].append(i)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="evaluate the final test (allowed once per version)")
    ap.add_argument("--data", default="v36", help="dataset version: v36 or v37")
    ap.add_argument("--force", action="store_true", help="re-run --final even if already touched")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()
    v = args.data
    final_lock = RESULTS_DIR / f"final_test_{v}_touched.json"
    args.models = [m.format(data=v) for m in args.models]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.final and final_lock.exists() and not args.force:
        raise SystemExit(f"final_test_{v} was already evaluated ({final_lock}). "
                         "Report that result; re-running turns it into a dev set.")

    train, val = load(f"train_{v}"), load(f"val_{v}")
    test_name = f"final_test_{v}" if args.final else f"heldout_test_{v}"
    test = load(test_name)
    probes = {"diagnostic_templates_v36": load("diagnostic_templates_v36"),
              "ambiguous_v36": load("ambiguous_v36")}
    print(f"train {len(train)} | val {len(val)} | {test_name} {len(test)}")

    scorers = {"style_lr": fit_style(train), "tfidf_lr": fit_tfidf(train)}
    for m in args.models:
        if (Path(m) / "config.json").exists():
            scorers[f"distilbert:{m}"] = load_distilbert(m)
        else:
            print(f"skip {m}: no checkpoint")

    val_labels = np.array([r["label"] for r in val])
    summary = {"test_set": test_name, "timestamp": datetime.now(timezone.utc).isoformat(),
               "bootstrap": {"n": N_BOOT, "unit": "group"}, "scorers": {}}
    per_sample = defaultdict(dict)
    test_slices = slices(test)

    for name, fn in scorers.items():
        print(f"\nScoring {name} ...", flush=True)
        rng = np.random.default_rng(0)
        vp = score_rows(fn, val)
        taus = choose_thresholds(val, vp)
        tp = score_rows(fn, test)
        res = {"thresholds": taus, "val_auroc": auroc(val_labels, vp), "slices": {}}
        for sl, idx in sorted(test_slices.items()):
            res["slices"][sl] = evaluate_slice([test[i] for i in idx], tp[idx], taus, rng)
        for pname, prow in probes.items():
            if prow:
                pp = score_rows(fn, prow)
                res["slices"][f"probe:{pname}"] = evaluate_slice(prow, pp, taus, rng)
        summary["scorers"][name] = res
        for i, p in enumerate(tp):
            per_sample[i][name] = round(float(p), 6)

        a = res["slices"]["ALL"]
        b = res["slices"].get("slice:source_balanced", {})
        f5 = a["at_val_fpr5"]
        print(f"  AUROC all={a['auroc']} balanced={b.get('auroc')} | @val_fpr5 tau={taus['val_fpr5']} "
              f"recall={f5.get('recall', {}).get('value')} fpr={f5.get('fpr', {}).get('value')}")

    out_json = RESULTS_DIR / f"{v}_eval_{'final' if args.final else 'heldout'}.json"
    out_jsonl = RESULTS_DIR / f"{v}_eval_{'final' if args.final else 'heldout'}_samples.jsonl"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, r in enumerate(test):
            f.write(json.dumps({"idx": i, "source": r["source"], "group": r["group"], "label": r["label"],
                                "text": r["text"][:300], "scores": per_sample[i]}, ensure_ascii=False) + "\n")

    print_table(summary)
    print_claim(summary)
    if args.final:
        final_lock.write_text(json.dumps({"touched_at": summary["timestamp"],
                                          "scorers": list(scorers)}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json} and {out_jsonl}")


def print_table(summary):
    names = list(summary["scorers"])
    short = [n.split("/")[-1] if n.startswith("distilbert") else n for n in names]
    keys = sorted(summary["scorers"][names[0]]["slices"])
    print("\nAUROC | recall@val_fpr5 | fpr@val_fpr5   (per slice)")
    print(f"{'slice':45s}" + "".join(f"{s[:24]:>26s}" for s in short))
    for k in keys:
        cells = []
        for n in names:
            s = summary["scorers"][n]["slices"][k]
            f5 = s["at_val_fpr5"]
            rec = f5.get("recall", {}).get("value")
            fpr = f5.get("fpr", {}).get("value")
            cell = f"{s['auroc'] if s['auroc'] is not None else '-'}|{rec if rec is not None else '-'}|{fpr if fpr is not None else '-'}"
            cells.append(f"{cell:>26s}")
        print(f"{k[:45]:45s}" + "".join(cells))


# Real-world headline: recall on real attacker-written indirect injections, FPR on real benign documents.
CLAIM_RECALL = ["source:rag_chunk_attacked_wasp_reddit_final", "source:reddit_wasp_attacked", "source:wasp_standalone",
                "source:rag_chunk_attacked_mixed_test", "source:satml_test", "source:goalswap_doc_test",
                "source:llmail_p2_api_triggered_test", "source:llmail_p2_judge_labeled_test",
                "source:llmail_context_attacked_test", "source:injecagent_test", "source:pubmed_final",
                "source:llmail_attack_api_triggered", "source:llmail_attack_judge_labeled",
                "source:llmail_context_attacked", "source:injecagent_dh", "source:injecagent_ds",
                "source:bbc_news_final", "slice:doc_injected", "slice:long_documents"]
CLAIM_FPR = ["source:rag_chunk_clean_wasp_reddit_final", "source:reddit_benign", "source:rag_chunk_clean_mixed_test",
             "source:goalswap_doc_test", "source:llmail_benign_email_test", "source:llmail_context_benign_test", "source:injecagent_benign_twin_test",
             "source:bbc_news_test", "source:no_robots_test", "source:pubmed_final",
             "source:llmail_benign_email", "source:llmail_context_benign", "source:bbc_news_final",
             "source:no_robots", "slice:meta_discussion", "slice:persona_benign", "slice:long_documents"]


def fmt_ci(d):
    return f"{d['value']:.3f} [{d['ci95'][0]:.3f}-{d['ci95'][1]:.3f}]" if d else "-"


def print_claim(summary):
    for name, res in summary["scorers"].items():
        sl = res["slices"]
        lines = []
        for metric, keys in (("recall", CLAIM_RECALL), ("fpr", CLAIM_FPR)):
            for k in keys:
                if k in sl:
                    for t in ("val_fpr5", "val_fpr1"):
                        d = sl[k][f"at_{t}"].get(metric)
                        if d:
                            lines.append(f"    {metric:6s} {k[:42]:42s} @{t}: {fmt_ci(d)}")
        if lines:
            print(f"\nREAL-WORLD HEADLINE  {name}")
            print("\n".join(lines))


if __name__ == "__main__":
    main()
