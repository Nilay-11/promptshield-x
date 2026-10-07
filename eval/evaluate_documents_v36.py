"""
PromptShield X - Phase 3.6 end-to-end evaluation on rendered real-world documents.

Renders every spec in format_suite_v36_{heldout,final}.json to an actual PDF (PyMuPDF) or HTML
page, runs the app's own extractors (PDFExtractor / WebExtractor), scores every extracted
segment and takes the document score as the max over segments (long segments use sliding
windows). Attacks are hidden with the same mechanisms as the benign controls (micro-font
footnotes, white text, off-canvas, metadata, display:none, comments, aria-hidden, zero-width),
so structural signals alone cannot separate the classes.

Scorers: TF-IDF baseline and each DistilBERT checkpoint. The document-level threshold
(FPR <= 5% / 1%) is chosen on format_suite_v36_val, rendered from val documents with val
attacks, and applied unchanged to the heldout / final suite. For PDFs, the production DocumentScanner pipeline
(rules + model + structural amplifier, current production weights) is also reported, with
"flagged" meaning overall_action != PASS.

Usage:
    python eval/evaluate_documents_v36.py            # heldout suite
    python eval/evaluate_documents_v36.py --final    # final suite, allowed ONCE
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import fitz
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (PROJECT_ROOT, PROJECT_ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_v36 import (DATA_DIR, DEFAULT_MODELS, RESULTS_DIR, choose_thresholds, cluster_boot, fit_tfidf, load,
                          load_distilbert, score_rows, threshold_at_fpr)
from app.modules.extractors.pdf_extractor import PDFExtractor
from app.modules.extractors.web_extractor import WebExtractor

ZW = "​"


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_pdf(spec) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    body, payload, m = spec["body"], spec["payload"], spec["method"]
    rect = fitz.Rect(50, 60, 545, 700)
    if m == "visible":
        sents = body.split(". ")
        k = max(1, len(sents) // 2)
        body = ". ".join(sents[:k]) + ". " + payload + " " + ". ".join(sents[k:])
    page.insert_textbox(rect, body, fontsize=10)
    if m == "microfont":
        page.insert_textbox(fitz.Rect(50, 760, 545, 800), payload, fontsize=1.2)
    elif m == "white_text":
        page.insert_textbox(fitz.Rect(50, 710, 545, 790), payload, fontsize=9, color=(1, 1, 1))
    elif m == "off_canvas":
        page.insert_text((50, -40), payload, fontsize=9)
    elif m == "metadata":
        doc.set_metadata({"title": payload[:500], "subject": payload[:500], "author": "Document Services"})
    data = doc.tobytes()
    doc.close()
    return data


def render_html(spec) -> bytes:
    paras = [p for p in spec["body"].replace("\n", " ").split(". ") if p.strip()]
    k = max(1, len(paras) // 2)
    payload, m = spec["payload"], spec["method"]
    esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if m == "visible":
        hidden = f"<p>{esc(payload)}</p>"
    elif m == "display_none":
        hidden = f'<div style="display:none">{esc(payload)}</div>'
    elif m == "html_comment":
        hidden = f"<!-- {payload.replace('--', '- -')} -->"
    elif m == "aria_hidden":
        hidden = f'<span aria-hidden="true">{esc(payload)}</span>'
    else:  # zero_width: payload visible but interleaved with zero-width spaces
        hidden = f"<p>{esc(ZW.join(payload))}</p>"
    html = ["<html><head><title>Document</title></head><body><article>"]
    html += [f"<p>{esc(p)}.</p>" for p in paras[:k]] + [hidden] + [f"<p>{esc(p)}.</p>" for p in paras[k:]]
    html.append("</article></body></html>")
    return "\n".join(html).encode("utf-8")


def extract_segments(spec, pdf_x, web_x):
    if spec["format"] == "pdf":
        res = pdf_x.extract(render_pdf(spec), filename="doc.pdf")
    else:
        res = web_x.extract(render_html(spec), filename="doc.html")
    return [s.content.strip() for s in res.segments if s.content and s.content.strip()]


def production_pdf_flags(specs):
    from app.modules.document_scanner import DocumentScanner

    scanner = DocumentScanner()
    flags = []
    for s in specs:
        if s["format"] != "pdf":
            flags.append(None)
            continue
        flags.append(scanner.scan_pdf_bytes(render_pdf(s), filename="doc.pdf")["overall_action"] != "PASS")
    return flags


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def document_scores(fn, specs, segs):
    """Max over extracted segments; long segments use sliding windows."""
    flat = [{"text": t, "is_long": len(t) > 800} for sg in segs for t in sg]
    owner = [i for i, sg in enumerate(segs) for _ in sg]
    out = np.zeros(len(specs))
    for i, v in zip(owner, score_rows(fn, flat)):
        out[i] = max(out[i], v)
    return out


def report(specs, scores, tau, rng):
    labels = np.array([s["label"] for s in specs])
    groups = np.array([s["group"] for s in specs])
    out = {}
    keys = defaultdict(list)
    for i, s in enumerate(specs):
        keys["ALL"].append(i)
        keys[f"format:{s['format']}"].append(i)
        keys[f"method:{s['format']}/{s['method']}"].append(i)
        keys[f"genre:{s['genre']}"].append(i)
    for k, idx in sorted(keys.items()):
        idx = np.array(idx)
        out[k] = {"n": len(idx), **cluster_boot(labels[idx], scores[idx], groups[idx], tau, rng)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--no-production", action="store_true", help="skip the DocumentScanner pipeline")
    ap.add_argument("--data", default="v36", help="dataset version: v36 or v37")
    args = ap.parse_args()
    v = args.data
    final_lock = RESULTS_DIR / f"format_suite_{v}_final_touched.json"
    args.models = [m.format(data=v) for m in args.models]

    if args.final and final_lock.exists() and not args.force:
        raise SystemExit(f"format_suite_{v}_final was already evaluated ({final_lock}).")
    suite = f"format_suite_{v}_final" if args.final else f"format_suite_{v}_heldout"
    specs = json.loads((DATA_DIR / f"{suite}.json").read_text(encoding="utf-8"))
    train, val = load(f"train_{v}"), load(f"val_{v}")
    print(f"{suite}: {len(specs)} rendered documents")

    val_specs = json.loads((DATA_DIR / f"format_suite_{v}_val.json").read_text(encoding="utf-8"))
    pdf_x, web_x = PDFExtractor(), WebExtractor()
    segs = [extract_segments(s, pdf_x, web_x) for s in specs]
    val_segs = [extract_segments(s, pdf_x, web_x) for s in val_specs]
    payload_found = np.mean([any(s["payload"][:40].lower() in x.lower().replace(ZW, "") for x in sg)
                             for s, sg in zip(specs, segs)])
    print(f"extractors surfaced the payload text in {payload_found:.1%} of documents")

    scorers = {"tfidf_lr": fit_tfidf(train)}
    for m in args.models:
        if (Path(m) / "config.json").exists():
            scorers[f"distilbert:{m}"] = load_distilbert(m)

    summary = {"suite": suite, "timestamp": datetime.now(timezone.utc).isoformat(),
               "payload_surfaced_by_extractors": round(float(payload_found), 4), "scorers": {}}
    val_labels = np.array([s["label"] for s in val_specs])
    for name, fn in scorers.items():
        print(f"\nScoring {name} ...", flush=True)
        val_scores = document_scores(fn, val_specs, val_segs)
        taus = {t: threshold_at_fpr(val_labels, val_scores, f) for t, f in (("val_fpr5", 0.05), ("val_fpr1", 0.01))}
        # second operating point that does not depend on the rendered val suite: the text-level
        # val_v36 FPR<=1% threshold, applied per segment
        taus["text_val_fpr1"] = choose_thresholds(val, score_rows(fn, val))["val_fpr1"]["short"]
        scores = document_scores(fn, specs, segs)
        summary["scorers"][name] = {
            "doc_thresholds": taus,
            "results": {t: report(specs, scores, tau, np.random.default_rng(0)) for t, tau in taus.items()}}

    if not args.no_production:
        print("\nRunning production DocumentScanner on PDFs ...", flush=True)
        flags = production_pdf_flags(specs)
        idx = [i for i, f in enumerate(flags) if f is not None]
        pdf_specs = [specs[i] for i in idx]
        scores = np.array([1.0 if flags[i] else 0.0 for i in idx])
        summary["scorers"]["production_document_scanner"] = {
            "doc_thresholds": {"val_fpr5": "overall_action != PASS"},
            "results": {"val_fpr5": report(pdf_specs, scores, 0.5, np.random.default_rng(0))}}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{suite}_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nrecall | fpr per slice at the val FPR<=5% threshold")
    names = list(summary["scorers"])
    print(f"{'slice':34s}" + "".join(f"{n.split('/')[-1][:22]:>24s}" for n in names))
    all_keys = sorted({k for n in names for k in summary["scorers"][n]["results"]["val_fpr5"]})
    for k in all_keys:
        cells = []
        for n in names:
            r = summary["scorers"][n]["results"]["val_fpr5"].get(k)
            if not r:
                cells.append(f"{'':>24s}")
                continue
            rec = r.get("recall", {}).get("value", "-")
            fpr = r.get("fpr", {}).get("value", "-")
            cells.append(f"{f'{rec}|{fpr}':>24s}")
        print(f"{k[:34]:34s}" + "".join(cells))
    if args.final:
        final_lock.write_text(json.dumps({"touched_at": summary["timestamp"]}), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
