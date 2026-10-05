"""
PromptShield X - Extractor Package
Unified abstractions and data structures for multi-vector indirect prompt injection defense.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SourceType(str, Enum):
    PDF = "pdf"
    SPREADSHEET = "spreadsheet"
    IMAGE = "image"
    WEB = "web"
    CODE = "code"
    DOCUMENTATION = "documentation"


class ThreatCategory(str, Enum):
    HIDDEN_TEXT = "hidden_text"
    METADATA_INJECTION = "metadata_injection"
    FORMULA_INJECTION = "formula_injection"
    VISUAL_PROMPT_INJECTION = "visual_prompt_injection"
    DOM_CLOAKING = "dom_cloaking"
    TROJAN_SOURCE_UNICODE = "trojan_source_unicode"
    COMMENT_HIJACK = "comment_hijack"
    ZERO_WIDTH_OBSCURITY = "zero_width_obscurity"
    DIRECT_INJECTION = "direct_injection"


@dataclass
class ExtractedSegment:
    """Represents an atomic text segment extracted from any document or artifact."""
    content: str
    source_type: SourceType
    location: Any  # e.g. "Page 3", or dict {"sheet": "Employees", "cell": "B17"}
    is_hidden: bool = False
    segment_type: str = "DOCUMENT_TEXT"  # DOCUMENT_TEXT | PDF_METADATA | CLOAKED_TEXT | LAYOUT_ANOMALY | ANNOTATION
    cloaking_signal: str = "NONE"        # NONE | MICRO_FONT | WHITE_TEXT | OFF_CANVAS | METADATA_PAYLOAD
    layout_anomaly_score: float = 0.0
    visibility_score: float = 0.0
    position_anomaly_score: float = 0.0
    metadata_anomaly_score: float = 0.0
    threat_indicators: List[ThreatCategory] = field(default_factory=list)
    confidence_penalty: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source_type": self.source_type.value,
            "location": self.location,
            "is_hidden": self.is_hidden,
            "segment_type": self.segment_type,
            "cloaking_signal": self.cloaking_signal,
            "layout_anomaly_score": self.layout_anomaly_score,
            "visibility_score": self.visibility_score,
            "position_anomaly_score": self.position_anomaly_score,
            "metadata_anomaly_score": self.metadata_anomaly_score,
            "threat_indicators": [t.value for t in self.threat_indicators],
            "confidence_penalty": self.confidence_penalty,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """Consolidated result returned by any extractor."""
    source_type: SourceType
    filename: Optional[str]
    segments: List[ExtractedSegment] = field(default_factory=list)
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    raw_character_count: int = 0
    anomalies_detected: int = 0
    extraction_warnings: List[str] = field(default_factory=list)


class BaseExtractor(ABC):
    """Abstract base class for all PromptShield X data extractors."""

    @abstractmethod
    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        """Extract content into structured segments and identify structural cloaking."""
        pass