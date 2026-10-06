"""
PromptShield X — Central Configuration & Ablation Flags.
All behavior changes, thresholds, and modes are controlled here.
"""

from typing import Literal
from pydantic import BaseModel


class FirewallSettings(BaseModel):
    # Classifier operating mode
    # - "model_only": pure raw neural model probability (no rule blending)
    # - "rules_only": pure regex pattern scanner (no neural inference)
    # - "combined": hybrid neural + rule evidence synthesis (default)
    # - "semantic_only": ignore physical layout/structural PDF signals
    # - "structural_only": risk derived strictly from structural layout anomalies
    classifier_mode: Literal["combined", "model_only", "rules_only", "semantic_only", "structural_only"] = "combined"

    # Legacy override switch (default False for honest academic integrity)
    # If True, reproduces Override Paths 1, 2, 3 where regex replaces model output
    overrides_enabled: bool = False

    # Configurable Tiers (Hand-set thresholds until Phase 4)
    semantic_tier_low: int = 25    # Below this: semantic intent is safe
    semantic_tier_high: int = 60   # Above this: definite semantic attack
    structural_amplifier_max: float = 0.8  # Max boost structure can apply to semantic score: risk = sem * (1 + amp)
    ambiguous_tier_can_block: bool = False  # If False, ambiguous tier (25 < semantic < 60) caps at REVIEW, never BLOCK
    
    # Review & Block action thresholds
    review_risk_threshold: int = 30  # Min score for REVIEW/REWRITE
    block_risk_threshold: int = 60   # Min score for BLOCK

    # Structural Anomaly rules
    additive_hidden_penalty_enabled: bool = False  # Legacy +30 / floor-88 rule (default False)

    # Non-prose skipping guards
    skip_non_prose_enabled: bool = True
    min_prose_token_count: int = 3

    # PDF Layout Thresholds
    min_visible_font_size: float = 2.0

    # Reading Order Multi-Segment Window Pass
    reading_order_window_enabled: bool = True
    reading_order_window_size: int = 3

    # RAG heuristic ablation flag (default False)
    rag_pure_injection_block_enabled: bool = False

    # Phase 2: Input Normalization Ablation Flags
    normalizer_enabled: bool = True
    norm_unicode_nfkc: bool = True
    norm_zero_width: bool = True
    norm_homoglyphs: bool = True
    norm_leetspeak: bool = True
    norm_spaced_text: bool = True
    norm_base64: bool = True
    norm_rot13: bool = True
    norm_fuzzy_canonical: bool = True

    # Phase 3.5: Hidden-Text Review Path & Quoted Context
    hidden_text_review_p_threshold: float = 0.15
    quoted_context_rule_softening: bool = True


# Global singleton settings instance
settings = FirewallSettings()
