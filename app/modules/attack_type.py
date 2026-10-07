"""
PromptShield X - attack type: DIRECT / INDIRECT / JAILBREAK / PROMPT_EXTRACTION.

INDIRECT is decided by the input channel: an attack that arrives inside third-party content (PDF, RAG chunk,
web page, tool output, email) is indirect by definition. For every channel, a trained DistilBERT classifier
(eval/train_type_classifier.py; test accuracy 94%, macro-F1 0.91) names the technique: a direct instruction
override, a jailbreak, or prompt/secret extraction. For a user prompt the technique is the attack type.
"""

import re
from pathlib import Path
from typing import Any, Dict

JAILBREAK_MARKERS = re.compile(
    r"\b(dan|do anything now|developer mode|god mode|jailbr(?:oken|eak)|no (?:rules|restrictions|filters)|"
    r"unrestricted|unfiltered|uncensored)\b", re.I)

TYPE_DIR = Path(__file__).resolve().parent / "weights" / "type_classifier"
INDIRECT_CHANNELS = {"pdf", "rag", "web", "tool", "email", "document"}

_tok = None
_model = None


def _load():
    global _tok, _model
    if _model is None and (TYPE_DIR / "config.json").exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(str(TYPE_DIR), local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(str(TYPE_DIR), local_files_only=True).eval()
        _model.to("cuda" if torch.cuda.is_available() else "cpu")
    return _tok, _model


def classify_attack_type(text: str, channel: str = "user") -> Dict[str, Any]:
    """Call only for text already flagged as an attack. channel: "user" or one of INDIRECT_CHANNELS."""
    technique, confidence = "UNKNOWN", 0.0
    tok, model = _load()
    if model is not None and text.strip():
        import torch
        enc = tok(text, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)[0]
        i = int(probs.argmax())
        technique, confidence = model.config.id2label[i], round(float(probs[i]), 4)
        # Well-known jailbreak markers win over a DIRECT guess (short persona/mode jailbreaks are rarer in training).
        if technique == "DIRECT" and JAILBREAK_MARKERS.search(text):
            technique = "JAILBREAK"
    attack_type = "INDIRECT" if channel in INDIRECT_CHANNELS else technique
    return {"attack_type": attack_type, "technique": technique, "type_confidence": confidence}
