"""
PromptShield X - confusion matrices for the paper (2026-10-16).
(1) v40 document detector on the unseen final_test_v40: RAG chunks and whole test, strict / loose thresholds chosen on
    val_v40 at 1% / 5% FPR (same protocol as Tables II-III). (2) attack-type classifier (from type_classifier_results.json).
Output: eval/results/confusion_matrices.json and eval/results/confusion_matrices.png
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from evaluate_v36 import load, load_distilbert, threshold_at_fpr  # noqa: E402


def cm(y, pred):  # rows = actual (benign, attack), cols = predicted (benign, attack)
    return [[int(((y == 0) & ~pred).sum()), int(((y == 0) & pred).sum())],
            [int(((y == 1) & ~pred).sum()), int(((y == 1) & pred).sum())]]


def main():
    val = [r for r in load("val_v40") if not r.get("is_long")]
    final = load("final_test_v40")
    fn = load_distilbert(str(ROOT / "models/distilbert_v40"))
    pv, pf = fn([r["text"] for r in val]), fn([r["text"] for r in final])
    yv, yf = np.array([r["label"] for r in val]), np.array([r["label"] for r in final])
    chunk = np.array([r["source"].startswith(("rag_chunk_attacked_gentel", "rag_chunk_clean_gentel")) for r in final])
    out = {"binary_v40": {}}
    for op, t in (("strict", 0.01), ("loose", 0.05)):
        tau = threshold_at_fpr(yv, pv, t)
        out["binary_v40"][op] = {"RAG_chunks": cm(yf[chunk], pf[chunk] >= tau), "whole_test": cm(yf, pf >= tau)}
    tc = json.loads((ROOT / "eval/results/type_classifier_results.json").read_text())
    out["attack_type"] = {"labels": tc["labels"], "matrix": tc["test_confusion"]}
    (ROOT / "eval/results/confusion_matrices.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def draw(ax, m, labels, title):
        m = np.array(m); pct = m / m.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{m[i, j]:,}\n({pct[i, j]:.1%})", ha="center", va="center", fontsize=8,
                        color="white" if pct[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(labels)), labels, fontsize=8); ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8); ax.set_title(title, fontsize=9)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    draw(axes[0], out["binary_v40"]["loose"]["RAG_chunks"], ["Benign", "Attack"], "(a) Unseen RAG chunks, loose")
    draw(axes[1], out["binary_v40"]["strict"]["RAG_chunks"], ["Benign", "Attack"], "(b) Unseen RAG chunks, strict")
    draw(axes[2], tc["test_confusion"], ["Direct", "Jailbreak", "Prompt\nextr."], "(c) Attack-type classifier")
    fig.tight_layout(); fig.savefig(ROOT / "eval/results/confusion_matrices.png", dpi=300)


if __name__ == "__main__":
    main()
