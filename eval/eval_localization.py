"""
PromptShield X - source-localization accuracy (2026-10-15).
On the 1,000 attacked validation PDFs (format_suite_v36_val: visible, microfont, white_text, off_canvas, metadata;
200 each), for every PDF the pipeline BLOCKs, take the highest-risk flagged segment and check
  (a) span: most of its words (>= 50%) come from the injected payload rather than the benign body;
  (b) layer: its reported source layer matches where the payload was hidden
      (metadata -> PDF_METADATA; microfont / white_text / off_canvas -> CLOAKED_TEXT; visible -> DOCUMENT_TEXT).
Output: eval/results/localization.json
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from evaluate_documents_v36 import render_pdf  # noqa: E402
from app.modules.document_scanner import DocumentScanner  # noqa: E402

LAYER = {"metadata": {"PDF_METADATA"}, "microfont": {"CLOAKED_TEXT"}, "white_text": {"CLOAKED_TEXT"},
         "off_canvas": {"CLOAKED_TEXT"}, "visible": {"DOCUMENT_TEXT"}}
words = lambda t: re.findall(r"[a-z0-9']+", (t or "").lower())


def main():
    specs = [s for s in json.loads((ROOT / "eval/data/format_suite_v36_val.json").read_text(encoding="utf-8"))
             if s["format"] == "pdf" and str(s["label"]) == "1"]
    scanner = DocumentScanner()
    st = defaultdict(lambda: {"n": 0, "detected": 0, "span_ok": 0, "layer_ok": 0})
    for s in specs:
        m = st[s["method"]]; m["n"] += 1
        res = scanner.scan_pdf_bytes(render_pdf(s))
        if res["overall_action"] != "BLOCK" or not res["flagged_threats"]:
            continue
        m["detected"] += 1
        top = max(res["flagged_threats"], key=lambda t: t.get("final_risk", 0))
        tw, pw = words(top.get("full_text") or top.get("text_snippet")), set(words(s["payload"]))
        if tw and sum(w in pw for w in tw) / len(tw) >= 0.5:
            m["span_ok"] += 1
        if top.get("segment_type") in LAYER[s["method"]]:
            m["layer_ok"] += 1
    tot = {k: sum(v[k] for v in st.values()) for k in ("n", "detected", "span_ok", "layer_ok")}
    out = {"by_method": {k: {**v, "span_acc": round(v["span_ok"] / max(v["detected"], 1), 4),
                             "layer_acc": round(v["layer_ok"] / max(v["detected"], 1), 4)} for k, v in st.items()},
           "overall": {**tot, "span_acc": round(tot["span_ok"] / tot["detected"], 4), "layer_acc": round(tot["layer_ok"] / tot["detected"], 4)}}
    (ROOT / "eval/results/localization.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(out))


if __name__ == "__main__":
    main()
