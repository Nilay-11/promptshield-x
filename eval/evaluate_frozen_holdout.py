"""
Evaluates the Frozen Holdout (200 items: 100 attacks, 100 benign controls).
Reports per-perturbation:
- Rules-only hit rate (on attacks)
- Model-only hit rate (on attacks)
- Combined hit rate (on attacks)
- False-positive rate (on benign controls) for each mode
"""

import json
from collections import defaultdict
from app.core.settings import settings
from app.modules.distilbert_classifier import classify_prompt_distilbert

def run_evaluation():
    with open("eval/frozen_holdout.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    modes = ["rules_only", "model_only", "combined"]
    # metrics[mode][perturbation][label] = {"total": 0, "detected": 0}
    stats = {m: defaultdict(lambda: {"total": 0, "detected": 0}) for m in modes}

    for mode in modes:
        settings.classifier_mode = mode
        settings.overrides_enabled = False
        settings.normalizer_enabled = True

        for item in dataset:
            text = item["text"]
            pert = item["perturbation"]
            label = item["label"]

            res = classify_prompt_distilbert(text)
            verdict = res.get("verdict", "BENIGN")
            score = res.get("final_score", 0)
            is_flagged = (verdict == "INJECTION" or score >= 50)

            stats[mode][(pert, label)]["total"] += 1
            if is_flagged:
                stats[mode][(pert, label)]["detected"] += 1

    # Compile table
    perturbations = ["none", "homoglyph", "leetspeak", "typoglycemia", "zero_width", "spacing", "base64", "rot13", "fullwidth", "multilingual"]
    rows = []

    for pert in perturbations:
        # Attack recall (hit rate)
        r_rules = stats["rules_only"][(pert, "ATTACK")]["detected"] / stats["rules_only"][(pert, "ATTACK")]["total"]
        r_model = stats["model_only"][(pert, "ATTACK")]["detected"] / stats["model_only"][(pert, "ATTACK")]["total"]
        r_comb = stats["combined"][(pert, "ATTACK")]["detected"] / stats["combined"][(pert, "ATTACK")]["total"]

        # Benign False Positive Rate
        fpr_rules = stats["rules_only"][(pert, "BENIGN")]["detected"] / stats["rules_only"][(pert, "BENIGN")]["total"]
        fpr_model = stats["model_only"][(pert, "BENIGN")]["detected"] / stats["model_only"][(pert, "BENIGN")]["total"]
        fpr_comb = stats["combined"][(pert, "BENIGN")]["detected"] / stats["combined"][(pert, "BENIGN")]["total"]

        rows.append({
            "perturbation": pert,
            "rules_hit_rate": round(r_rules * 100, 1),
            "model_hit_rate": round(r_model * 100, 1),
            "combined_hit_rate": round(r_comb * 100, 1),
            "rules_fpr": round(fpr_rules * 100, 1),
            "model_fpr": round(fpr_model * 100, 1),
            "combined_fpr": round(fpr_comb * 100, 1),
        })

    # Overall totals
    total_atk = sum(stats["combined"][(p, "ATTACK")]["total"] for p in perturbations)
    total_ben = sum(stats["combined"][(p, "BENIGN")]["total"] for p in perturbations)

    overall = {
        "perturbation": "**OVERALL AVERAGE**",
        "rules_hit_rate": round(sum(stats["rules_only"][(p, "ATTACK")]["detected"] for p in perturbations) / total_atk * 100, 1),
        "model_hit_rate": round(sum(stats["model_only"][(p, "ATTACK")]["detected"] for p in perturbations) / total_atk * 100, 1),
        "combined_hit_rate": round(sum(stats["combined"][(p, "ATTACK")]["detected"] for p in perturbations) / total_atk * 100, 1),
        "rules_fpr": round(sum(stats["rules_only"][(p, "BENIGN")]["detected"] for p in perturbations) / total_ben * 100, 1),
        "model_fpr": round(sum(stats["model_only"][(p, "BENIGN")]["detected"] for p in perturbations) / total_ben * 100, 1),
        "combined_fpr": round(sum(stats["combined"][(p, "BENIGN")]["detected"] for p in perturbations) / total_ben * 100, 1),
    }
    rows.append(overall)

    with open("eval/frozen_holdout_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("Frozen holdout evaluation complete.")
    return rows

if __name__ == "__main__":
    run_evaluation()
