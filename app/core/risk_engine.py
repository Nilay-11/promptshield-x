"""
Risk Scoring Engine (Chapter 6.9) — simplified version for the zero-shot path.

Full version (per config.yaml risk_scoring.weights) combines pattern_severity,
classifier_confidence, chunk_risk, reliability_score, and anomaly_score. Until
the RAG-side modules (chunk scanner, anomaly detector, reliability filter)
are built, this uses just the two signals that exist today: the pattern
scanner's severity and the semantic classifier's confidence.
"""

CATEGORY_BASE_SEVERITY = {
    # 4-Class Taxonomy (DistilBERT Fine-Tuned)
    "BENIGN": 0,
    "DIRECT": 50,
    "INDIRECT": 50,
    "JAILBREAK_PROMPT_LEAKAGE": 90,
    # Zero-Shot / Legacy taxonomy fallback
    "safe": 0,
    "prompt_injection": 70,
    "jailbreak": 90,
    "prompt_extraction": 75,
    "agent_manipulation": 65,
}


def compute_risk_score(pattern_severity: int, classification: dict) -> dict:
    """
    pattern_severity: 0-100, max severity hit from the regex/keyword scanner
                       (0 if no rule matched)
    classification: output of semantic_classifier / distilbert_classifier

    Returns: {"risk_score": int, "category": str, "action": "PASS"|"REWRITE"|"BLOCK"}
    """
    category = classification.get("category", "BENIGN")
    confidence = float(classification.get("confidence", 0.0))
    raw_scores = classification.get("raw_scores", {})

    is_safe = (category in ["safe", "BENIGN"])
    base = CATEGORY_BASE_SEVERITY.get(category, 60)

    if is_safe:
        # If classifier says safe, pattern scanner severity still takes precedence
        benign_conf = max(confidence, raw_scores.get("BENIGN", raw_scores.get("safe", 0.0)))
        classifier_risk = (1.0 - benign_conf) * 50
        risk_score = max(pattern_severity, round(classifier_risk))
    else:
        # Classifier flagged an attack category
        benign_prob = raw_scores.get("BENIGN", raw_scores.get("safe", 0.0))
        attack_mass = max(confidence, 1.0 - benign_prob)
        classifier_signal = base * attack_mass

        if pattern_severity > 0:
            risk_score = round(0.4 * pattern_severity + 0.6 * classifier_signal)
        else:
            risk_score = round(classifier_signal)

    risk_score = max(0, min(100, risk_score))

    if risk_score <= 30:
        action = "PASS"
    elif risk_score <= 65:
        action = "REWRITE"
    else:
        action = "BLOCK"

    return {"risk_score": risk_score, "category": category, "action": action}


