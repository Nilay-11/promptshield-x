"""
PromptShield X - Semantic Anomaly Detector.
Uses Isolation Forest trained on benign vector embeddings (all-MiniLM-L6-v2)
to detect out-of-distribution adversarial vectors and cloaked prompt embeddings.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
MODEL_PATH = WEIGHTS_DIR / "isolation_forest.pkl"

logger = logging.getLogger(__name__)

_encoder = None
_isolation_forest = None

def _get_models():
    global _encoder, _isolation_forest
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer: {e}")
            _encoder = None

    if _isolation_forest is None and MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                _isolation_forest = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load Isolation Forest from {MODEL_PATH}: {e}")
            _isolation_forest = None

    return _encoder, _isolation_forest

def detect_anomaly(text: str) -> Dict[str, Any]:
    """
    Computes semantic anomaly score for input prompt.
    Returns:
        {
            "anomaly_score": int (0-100),
            "is_anomalous": bool,
            "source": "isolation_forest" | "fallback"
        }
    """
    encoder, forest = _get_models()
    if encoder is not None and forest is not None:
        try:
            embedding = encoder.encode([text], normalize_embeddings=True)
            # predict returns 1 for inlier, -1 for outlier
            pred = forest.predict(embedding)[0]
            score_raw = forest.decision_function(embedding)[0]
            
            # Map decision function (-0.5 to +0.5) to 0-100 scale where higher is more anomalous
            # Inlier yields positive score (e.g. +0.2 -> risk ~30), Outlier yields negative score (e.g. -0.2 -> risk ~70)
            norm_score = max(0, min(100, int((0.3 - score_raw) * 100)))
            is_anomalous = (pred == -1) or (norm_score > 60)

            return {
                "anomaly_score": norm_score,
                "is_anomalous": is_anomalous,
                "decision_score": round(float(score_raw), 4),
                "source": "isolation_forest"
            }
        except Exception as e:
            logger.warning(f"Anomaly inference error: {e}")

    # Fallback when weights or sentence transformers unavailable
    return {
        "anomaly_score": 25,
        "is_anomalous": False,
        "source": "fallback"
    }

if __name__ == "__main__":
    res = detect_anomaly("Calculate the average quarterly revenue from the database.")
    print("Anomaly Detection:", res)
