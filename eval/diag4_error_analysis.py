"""
Diagnostic 4: Error Analysis
Samples up to 50 false negatives and 50 false positives from deepset-test-only plus validation set.
Uses the production pipeline (classify_prompt_distilbert).
Saves per-sample scores to eval/results/diag4_error_analysis.jsonl and summary to eval/results/diag4_error_analysis.json.
Categorizes by:
- formal prose
- long text (> 100 words)
- quoted attack
- casual prompt
- agent-style / tool instruction
- non-English
- other
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from app.modules.distilbert_classifier import classify_prompt_distilbert
from app.core.settings import settings

settings.classifier_mode = "combined"

def categorize_error(text: str, is_fn: bool) -> str:
    words = text.split()
    tl = text.lower()
    
    # 1. Long text
    if len(words) > 100:
        return "long text"
    
    # 2. Quoted attack
    if re.search(r"[\x22\x27\x60\u201c\u201d\u2018\u2019].{5,}[\x22\x27\x60\u201c\u201d\u2018\u2019]", text):
        if any(w in tl for w in ["ignore", "disregard", "prompt", "dan", "reveal", "instruction"]):
            return "quoted attack"
            
    # 3. Agent-style / tool instruction
    if any(k in tl for k in ["tool", "api", "function", "json", "python", "script", "database", "sql", "email", "access", "send", "retrieve", "execute"]):
        if any(cmd in tl for cmd in ["grant", "fetch", "email them to", "dump", "unlock", "transfer", "payee", "search for"]):
            return "agent-style"
            
    # 4. Formal prose (enterprise, legal, financial, reports)
    if any(k in tl for k in ["footnote", "pursuant", "agreement", "confidential", "liability", "damages", "depreciation", "operating", "quarter", "compliance", "policy", "standards"]):
        return "formal prose"
        
    # 5. Casual prompt
    if any(k in tl for k in ["what is", "how do", "can you", "help me", "write a", "explain", "summarize", "recipe", "tell me"]):
        return "casual prompt"
        
    # 6. Non-English
    non_ascii = len([c for c in text if ord(c) > 127 and not (0x2018 <= ord(c) <= 0x201d)])
    if non_ascii > len(text) * 0.2:
        return "non-English"
        
    return "other"


def run_error_analysis():
    # Load validation and deepset test sets
    with open("eval/data/val_v35.json", encoding="utf-8") as f:
        val_data = json.load(f)
    with open("eval/data/heldout_test_v35.json", encoding="utf-8") as f:
        heldout_data = json.load(f)

    deepset_test = [s for s in heldout_data if s.get("source") == "deepset_test"]
    combined_pool = val_data + deepset_test
    print(f"Total evaluation pool: {len(combined_pool)} (Val: {len(val_data)}, Deepset-test: {len(deepset_test)})")

    fn_list = []
    fp_list = []

    for item in combined_pool:
        text = item["text"]
        true_label = int(item["label"])
        res = classify_prompt_distilbert(text)
        p_inj = res.get("model_p_inj", 0.0) or 0.0
        final_score = res["final_score"]
        pred_label = 1 if final_score >= 50 else 0

        err_item = {
            "text": text,
            "true_label": true_label,
            "pred_label": pred_label,
            "p_inj": p_inj,
            "final_score": final_score,
            "source": item.get("source", "unknown"),
            "length_words": len(text.split()),
            "length_chars": len(text),
        }

        if true_label == 1 and pred_label == 0:
            err_item["error_type"] = "FN"
            err_item["category"] = categorize_error(text, is_fn=True)
            fn_list.append(err_item)
        elif true_label == 0 and pred_label == 1:
            err_item["error_type"] = "FP"
            err_item["category"] = categorize_error(text, is_fn=False)
            fp_list.append(err_item)

    print(f"Total False Negatives found: {len(fn_list)}")
    print(f"Total False Positives found: {len(fp_list)}")

    sampled_fn = fn_list[:50]
    sampled_fp = fp_list[:50]

    fn_categories = {}
    for item in sampled_fn:
        fn_categories[item["category"]] = fn_categories.get(item["category"], 0) + 1

    fp_categories = {}
    for item in sampled_fp:
        fp_categories[item["category"]] = fp_categories.get(item["category"], 0) + 1

    # Save to JSONL and JSON
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "diag4_error_analysis.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in sampled_fn + sampled_fp:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_path = out_dir / "diag4_error_analysis.json"
    summary = {
        "pool_size": len(combined_pool),
        "total_fn_found": len(fn_list),
        "total_fp_found": len(fp_list),
        "sampled_fn_count": len(sampled_fn),
        "sampled_fp_count": len(sampled_fp),
        "fn_category_counts": fn_categories,
        "fp_category_counts": fp_categories,
        "sampled_false_negatives": sampled_fn,
        "sampled_false_positives": sampled_fp,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n--- ERROR ANALYSIS SUMMARY ---")
    print(f"Sampled False Negatives ({len(sampled_fn)}):")
    for cat, cnt in sorted(fn_categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {cnt:2d}")
    print(f"\nSampled False Positives ({len(sampled_fp)}):")
    for cat, cnt in sorted(fp_categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {cnt:2d}")
    print(f"\nSaved raw per-sample scores to {jsonl_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    run_error_analysis()
