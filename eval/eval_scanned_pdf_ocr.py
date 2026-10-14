"""
PromptShield X - scanned-PDF OCR evaluation (2026-10-15).
100 image-only ("scanned") one-page PDFs rendered from unseen final_test_v40 RAG chunks: 50 attacked, 50 clean.
Each chunk is drawn as an image (no text layer), so the only way to see it is OCR. Scored by the production
DocumentScanner; a PDF counts as detected when overall_action is BLOCK (and REVIEW-or-BLOCK separately).
Output: eval/results/scanned_pdf_ocr.json
"""
import io, json, sys, textwrap, time
from pathlib import Path
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.modules.document_scanner import DocumentScanner  # noqa: E402

FONT = ImageFont.truetype("arial.ttf", 22)


def scanned_pdf(text):
    im = Image.new("RGB", (1240, 1754), "white"); d = ImageDraw.Draw(im)
    for i, line in enumerate(textwrap.wrap(text, 95)[:60]):
        d.text((70, 80 + 28 * i), line, font=FONT, fill="black")
    b = io.BytesIO(); im.save(b, "PNG")
    doc = fitz.open(); p = doc.new_page(); p.insert_image(p.rect, stream=b.getvalue())
    return doc.tobytes()


def main():
    rows = json.loads((ROOT / "eval/data/final_test_v40.json").read_text(encoding="utf-8"))
    chunks = [r for r in rows if r["source"].startswith(("rag_chunk_attacked", "rag_chunk_clean")) and 200 < len(r["text"]) < 2500]
    rng = np.random.default_rng(0)
    att = [r for r in chunks if r["label"] == 1]; ben = [r for r in chunks if r["label"] == 0]
    sample = [att[i] for i in rng.choice(len(att), 50, replace=False)] + [ben[i] for i in rng.choice(len(ben), 50, replace=False)]
    scanner = DocumentScanner()
    y, block, flag, ts = [], [], [], []
    for r in sample:
        t0 = time.perf_counter(); res = scanner.scan_pdf_bytes(scanned_pdf(r["text"])); ts.append(time.perf_counter() - t0)
        y.append(r["label"]); block.append(res["overall_action"] == "BLOCK"); flag.append(res["overall_action"] in ("BLOCK", "REVIEW"))
    y, block, flag = map(np.array, (y, block, flag))
    out = {"n_attacked": int(y.sum()), "n_clean": int((y == 0).sum()),
           "block": {"recall": float(block[y == 1].mean()), "fpr": float(block[y == 0].mean())},
           "review_or_block": {"recall": float(flag[y == 1].mean()), "fpr": float(flag[y == 0].mean())},
           "mean_seconds_per_pdf": round(float(np.mean(ts)), 2)}
    (ROOT / "eval/results/scanned_pdf_ocr.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("RESULT", json.dumps(out))


if __name__ == "__main__":
    main()
