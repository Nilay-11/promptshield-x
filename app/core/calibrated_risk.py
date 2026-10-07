"""
PromptShield X - calibrated risk: risk = 100 x P(attack | signals), learned from validation data.

Signals: the prompt detector and (documents only) the document detector probabilities as logits, max rule severity,
rule-hit flag, log word count. A logistic score per channel (prompt / document) mapped to a probability by an isotonic
curve, both fitted on val_v40 by
eval/train_risk_calibrator.py and stored in weights/risk_calibrator.json together with the REVIEW / BLOCK
probability thresholds chosen on validation data at 2% / 0.5% false-positive rate.

Probabilities are prior-corrected to a stated attack rate (PS_ATTACK_PREVALENCE, default 10%): a calibrated probability
only means something relative to how common attacks are. Decisions use thresholds validated on data
(documents: calibrated thresholds; prompts: rule + model agreement), not the displayed number.
Measured on heldout_test_v40 (at the validation attack rate): ECE 0.058 for prompts (raw model 0.100) and 0.0075 for
documents (raw model 0.021). See eval/results/risk_calibration.json.
"""

import bisect
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

CALIBRATOR = Path(__file__).resolve().parent.parent / "modules" / "weights" / "risk_calibrator.json"


@lru_cache(maxsize=1)
def _params() -> Dict[str, Any]:
    return json.loads(CALIBRATOR.read_text(encoding="utf-8")) if CALIBRATOR.exists() else {}


def available() -> bool:
    return bool(_params())


def _model_prob(text: str, role: str) -> float:
    """Raw detector probability, scored exactly like the calibration data (512-token windows, max over windows)."""
    import torch
    from app.modules.distilbert_classifier import _get_distilbert
    tok, model = _get_distilbert(role)
    if model is None or not text.strip():
        return 0.0
    enc = tok(text, truncation=True, max_length=512, stride=256, return_overflowing_tokens=True,
              padding=True, return_tensors="pt")
    enc.pop("overflow_to_sample_mapping", None)
    with torch.no_grad():
        probs = torch.softmax(model(**{k: v.to(model.device) for k, v in enc.items()}).logits.float(), dim=-1)[:, 1]
    return float(probs.max())


def _isotonic(z: float, xs, ys) -> float:
    """Piecewise-linear isotonic map (same as sklearn IsotonicRegression.predict with clipping)."""
    if z <= xs[0]:
        return ys[0]
    if z >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, z)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 if x1 == x0 else y0 + (y1 - y0) * (z - x0) / (x1 - x0)


def prevalence() -> float:
    try:
        return min(max(float(os.environ.get("PS_ATTACK_PREVALENCE", "0.10")), 0.001), 0.999)
    except ValueError:
        return 0.10


def _prior_correct(p: float, fit_rate: float, target_rate: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    odds = p / (1 - p) * (target_rate / (1 - target_rate)) / (fit_rate / (1 - fit_rate))
    return odds / (1 + odds)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def score(text: str, rule_severity: int, channel: str) -> Dict[str, Any]:
    """channel: "prompt" (typed by a user) or "document" (file / web / RAG / tool content)."""
    cfg = _params()[channel]
    p_prompt = _model_prob(text, "prompt")
    p_doc = _model_prob(text, "document") if channel == "document" else 0.0  # prompts: prompt detector + rules only
    doc = channel == "document"  # prompts use the prompt detector + rules only (no document detector, no length)
    x = [_logit(p_prompt), _logit(p_doc) if doc else 0.0, rule_severity / 100.0, 1.0 if rule_severity > 0 else 0.0,
         math.log1p(len(text.split())) if doc else 0.0]
    z = cfg["intercept"] + sum(w * v for w, v in zip(cfg["coef"], x))
    prob_fit = _isotonic(z, cfg["iso_x"], cfg["iso_y"]) if cfg.get("iso_x") else 1 / (1 + math.exp(-z))
    # Bayes prior correction: the calibrator learned P(attack) at the validation set's attack rate (~50-60%);
    # real traffic has far fewer attacks. Shown risk = P(attack) at PS_ATTACK_PREVALENCE (default 10%).
    prob = _prior_correct(prob_fit, cfg.get("fit_attack_rate", 0.5), prevalence())
    return {"probability": round(prob, 4), "risk": int(round(100 * prob)), "channel": channel,
            "probability_at_fit_rate": round(prob_fit, 4), "assumed_attack_rate": prevalence(),
            "review_threshold": round(cfg["review_threshold"], 4), "block_threshold": round(cfg["block_threshold"], 4),
            "signals": {"p_prompt_model": round(p_prompt, 4), "p_document_model": round(p_doc, 4),
                        "rule_severity": rule_severity}}
