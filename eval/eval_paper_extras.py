"""
PromptShield X - additional measurements for the paper (2026-10-12).

1. Baselines on the v40 final test (the same data as Table II):
   no defense, keyword rules only, TF-IDF + LR, two published detectors (ProtectAI DeBERTa-v3 prompt-injection v2,
   deepset DeBERTa-v3 injection), and the PromptShield v40 detector. Thresholds for every learned scorer are chosen on
   val_v40 (1% / 5% FPR) and applied unchanged, exactly like the original evaluation.
2. Ablation on the validation rendered-document suite (2,000 PDF + 2,000 HTML; 50% attacked, 50% benign twins):
   full pipeline (source-specific extraction) vs text-only extraction (plain visible text, no hidden layers / metadata /
   markup scan), both scored by the same document detector and calibrated thresholds.
3. Latency of the full gateway (GPU: RTX 3050 Laptop) for prompts, CSV files and PDFs: mean and 95th percentile.

Output: eval/results/paper_extras.json
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))

from evaluate_v36 import cluster_boot, fit_tfidf, load, load_distilbert, threshold_at_fpr  # noqa: E402
from app.modules.pattern_scanner import scan_prompt  # noqa: E402

OUT = ROOT / "eval/results/paper_extras.json"
RNG = np.random.default_rng(0)


def published_scorer(name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(name, local_files_only=True).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    @torch.no_grad()
    def score(texts):
        out = np.zeros(len(texts))
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), 32):
            ids = order[s:s + 32]
            enc = tok([texts[i] for i in ids], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                logits = model(**enc).logits
            out[ids] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
        return out
    return score


def rate(pred, y, cls):
    m = y == cls
    return float(pred[m].mean()) if m.any() else None


def evaluate_baselines():
    val, final = load("val_v40"), load("final_test_v40")
    val = [r for r in val if not r.get("is_long")]
    yv = np.array([r["label"] for r in val])
    yf = np.array([r["label"] for r in final])
    gf = np.array([r["group"] for r in final])
    chunk_idx = np.array([r["source"].startswith(("rag_chunk_attacked_gentel", "rag_chunk_clean_gentel")) for r in final])
    scorers = {
        "tfidf_lr": fit_tfidf(load("train_v40")),
        "protectai_deberta_v3_v2": published_scorer("protectai/deberta-v3-base-prompt-injection-v2"),
        "deepset_deberta_v3": published_scorer("deepset/deberta-v3-base-injection"),
        "promptshield_v40": load_distilbert(str(ROOT / "models/distilbert_v40")),
    }
    vt, ft = [r["text"] for r in val], [r["text"] for r in final]
    res = {}
    # no defense: nothing is ever flagged
    res["no_defense"] = {"note": "forwards everything", "ALL": {"recall": 0.0, "fpr": 0.0}, "RAG_chunks": {"recall": 0.0, "fpr": 0.0}}
    # keyword rules only: flagged when any injection pattern matches (no threshold to tune)
    kw = np.array([scan_prompt(t).get("severity", 0) > 0 for t in ft])
    res["keyword_rules"] = {
        "ALL": {"recall": rate(kw, yf, 1), "fpr": rate(kw, yf, 0)},
        "RAG_chunks": {"recall": rate(kw[chunk_idx], yf[chunk_idx], 1), "fpr": rate(kw[chunk_idx], yf[chunk_idx], 0)},
    }
    for name, fn in scorers.items():
        print(f"scoring {name} ...", flush=True)
        pv, pf = fn(vt), fn(ft)
        from sklearn.metrics import roc_auc_score
        entry = {"val_auroc": round(float(roc_auc_score(yv, pv)), 4), "final_auroc": round(float(roc_auc_score(yf, pf)), 4),
                 "chunk_auroc": round(float(roc_auc_score(yf[chunk_idx], pf[chunk_idx])), 4)}
        for op, target in (("strict", 0.01), ("loose", 0.05)):
            tau = threshold_at_fpr(yv, pv, target)
            entry[op] = {"threshold": tau,
                         "ALL": cluster_boot(yf, pf, gf, tau, RNG),
                         "RAG_chunks": cluster_boot(yf[chunk_idx], pf[chunk_idx], gf[chunk_idx], tau, RNG)}
        res[name] = entry
        c = entry["loose"]["RAG_chunks"]
        print(f"   chunk AUROC {entry['chunk_auroc']} | loose RAG recall {c['recall']['value']} fpr {c['fpr']['value']}", flush=True)
    return res


def evaluate_ablation():
    import fitz
    from bs4 import BeautifulSoup
    from evaluate_documents_v36 import render_html, render_pdf
    from app.core.gateway import _doc_decision, _score_text
    from app.modules.document_scanner import DocumentScanner
    from app.modules.extractors.web_extractor import WebExtractor
    specs = json.loads((ROOT / "eval/data/format_suite_v36_val.json").read_text(encoding="utf-8"))
    scanner, web = DocumentScanner(), WebExtractor()
    out = {}
    for fmt in ("pdf", "html"):
        sp = [s for s in specs if s["format"] == fmt]
        y = np.array([s["label"] for s in sp])
        full, plain = [], []
        for s in sp:
            if fmt == "pdf":
                data = render_pdf(s)
                full.append(scanner.scan_pdf_bytes(data)["overall_action"] == "BLOCK")
                doc = fitz.open(stream=data, filetype="pdf")
                text = "\n".join(p.get_text() for p in doc); doc.close()
            else:
                html = render_html(s)
                segs = [x.content for x in web.extract(html).segments if x.content.strip()]
                full.append(any(_doc_decision(_score_text(t, "document")["action"], 0, True) == "BLOCK" for t in segs))
                text = BeautifulSoup(html, "html.parser").get_text(" ")  # visible-text extraction, comments dropped
            sc = _score_text(text, "document")
            plain.append(_doc_decision(sc["action"], sc["risk"], bool(sc.get("calibration"))) == "BLOCK")
        full, plain = np.array(full), np.array(plain)
        methods = sorted({s["method"] for s in sp})
        out[fmt] = {
            "n": len(sp),
            "full_pipeline": {"recall": rate(full, y, 1), "fpr": rate(full, y, 0)},
            "text_only": {"recall": rate(plain, y, 1), "fpr": rate(plain, y, 0)},
            "by_method": {m: {"full_recall": rate(full[[s["method"] == m for s in sp]], y[[s["method"] == m for s in sp]], 1),
                              "text_only_recall": rate(plain[[s["method"] == m for s in sp]], y[[s["method"] == m for s in sp]], 1)}
                          for m in methods},
        }
        print(fmt, out[fmt]["full_pipeline"], out[fmt]["text_only"], flush=True)
    return out


def evaluate_latency():
    from app.core import gateway
    from evaluate_documents_v36 import render_pdf
    prompts = [r["text"] for r in load("heldout_test_v40") if not r.get("is_long") and len(r["text"]) < 2000][:300]
    specs = [s for s in json.loads((ROOT / "eval/data/format_suite_v36_val.json").read_text(encoding="utf-8")) if s["format"] == "pdf"][:60]
    csv = b"name,note\n" + b"\n".join(f"row{i},Paid invoice {i} on time".encode() for i in range(40))
    gateway.process("warm up", user_id="latency")  # load models first
    def timed(fn, n):
        ts = []
        for i in range(n):
            t0 = time.perf_counter(); fn(i); ts.append((time.perf_counter() - t0) * 1000)
        return {"n": n, "mean_ms": round(statistics.mean(ts), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1)}
    return {
        "hardware": "NVIDIA RTX 3050 Laptop GPU (6 GB), mock LLM (security pipeline only)",
        "prompt": timed(lambda i: gateway.process(prompts[i % len(prompts)], user_id="latency"), 300),
        "csv_40_rows": timed(lambda i: gateway.process("Analyse this", user_id="latency", files=[("t.csv", csv)]), 30),
        "pdf_1_page": timed(lambda i: gateway.process("Summarize", user_id="latency", files=[("t.pdf", render_pdf(specs[i % len(specs)]))]), 60),
    }


def main():
    import os
    os.environ.setdefault("PS_LLM_PROVIDER", "mock")
    from app.core import gateway
    gateway.DB_PATH = Path(ROOT / "eval/results/_latency_tmp.db")  # keep measurement traffic out of the real audit DB
    res = {"baselines_v40_final": evaluate_baselines(), "ablation_validation_documents": evaluate_ablation(),
           "latency": evaluate_latency()}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    Path(ROOT / "eval/results/_latency_tmp.db").unlink(missing_ok=True)
    print(json.dumps(res["latency"], indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
