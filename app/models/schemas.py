from typing import Literal, Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    retrieved_chunks: Optional[list[str]] = None  # populated for /analyze-rag


class AnalyzeResponse(BaseModel):
    action: Literal["PASS", "REWRITE", "BLOCK"]
    risk_score: int  # 0-100, see config.yaml risk_scoring.thresholds
    category: Literal[
        "safe",
        "prompt_injection",
        "jailbreak",
        "prompt_extraction",
        "agent_manipulation",
    ]
    details: str
    rewritten_prompt: Optional[str] = None
    # Attack type (only when the input is flagged): DIRECT / INDIRECT / JAILBREAK / PROMPT_EXTRACTION
    attack_type: Optional[str] = None
    technique: Optional[str] = None  # DIRECT / JAILBREAK / PROMPT_EXTRACTION from the type classifier
    type_confidence: Optional[float] = None


# --- Added for /analyze-rag (Person B — minimal RAG path) ---

class ChunkRiskResult(BaseModel):
    index: int
    chunk_preview: str
    risk_score: int
    category: Literal[
        "safe",
        "prompt_injection",
        "jailbreak",
        "prompt_extraction",
        "agent_manipulation",
    ]
    action: Literal["PASS", "REWRITE", "BLOCK"]
    pattern_matches: list[str]  # rule ids from pattern_scanner, e.g. ["ignore_instructions_v1"]
    classifier_confidence: float
    # Attack type (only when the input is flagged): DIRECT / INDIRECT / JAILBREAK / PROMPT_EXTRACTION
    attack_type: Optional[str] = None
    technique: Optional[str] = None  # DIRECT / JAILBREAK / PROMPT_EXTRACTION from the type classifier
    type_confidence: Optional[float] = None


class AnalyzeRagResponse(BaseModel):
    total_chunks: int
    overall_risk_score: int
    overall_action: Literal["PASS", "REWRITE", "BLOCK"]
    chunks: list[ChunkRiskResult]