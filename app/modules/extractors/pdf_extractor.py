"""
PromptShield X - PDF Extractor
Extracts body text, document metadata, form fields, and annotations.
Detects cloaked indirect prompt injections:
- Microscopic / zero-point fonts (font size < 2.0pt)
- Off-canvas / negative coordinate text
- Invisible / white-on-white text (matching background)
- Covert payloads in PDF metadata & sticky note annotations
"""

import io
import sys
from pathlib import Path
from typing import Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.extractors import (
    BaseExtractor,
    ExtractedSegment,
    ExtractionResult,
    SourceType,
    ThreatCategory,
)


# Safe import: 'fitz' is installed via 'pip install pymupdf'
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    fitz = None
    HAS_PYMUPDF = False


class PDFExtractor(BaseExtractor):
    def __init__(self, min_visible_font_size: float = 2.0):
        """
        :param min_visible_font_size: Any text with font size below this threshold
                                      is flagged as hidden/adversarial cloaking.
        """
        self.min_visible_font_size = min_visible_font_size

    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        fname = filename or "document.pdf"
        result = ExtractionResult(source_type=SourceType.PDF, filename=fname)

        if not HAS_PYMUPDF:
            result.extraction_warnings.append(
                "PyMuPDF is not installed. Please run 'pip install pymupdf' to enable PDF extraction."
            )
            return result

        doc = None
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            import re

            FREE_TEXT_METADATA_KEYS = {"title", "subject", "keywords", "author"}
            SUSPICIOUS_METADATA_REGEX = re.compile(
                r"(?i)\b(ignore|disregard|system\s*prompt|you are now|dan|override|bypass|reveal)\b"
            )

            # ----------------------------------------------------
            # 1. Document Metadata Normalization (Phase 1, Item 5)
            # Metadata is NOT hidden text; no fixed penalties.
            # ----------------------------------------------------
            if doc.metadata:
                result.document_metadata = dict(doc.metadata)
                for meta_key, meta_val in doc.metadata.items():
                    if meta_val and isinstance(meta_val, str):
                        clean_val = meta_val.strip()
                        if meta_key in FREE_TEXT_METADATA_KEYS and len(clean_val) >= 20 \
                                and not SUSPICIOUS_METADATA_REGEX.search(clean_val):
                            # Free-text metadata always reaches the classifier: a keyword gate
                            # alone lets paraphrased injections through. No anomaly score here.
                            result.segments.append(
                                ExtractedSegment(
                                    content=clean_val,
                                    source_type=SourceType.PDF,
                                    location={"element": "metadata", "key": meta_key},
                                    segment_type="PDF_METADATA",
                                    metadata={"meta_key": meta_key},
                                )
                            )
                        # Only flag metadata as a threat if it contains an actual attack directive
                        if SUSPICIOUS_METADATA_REGEX.search(clean_val):
                            result.segments.append(
                                ExtractedSegment(
                                    content=clean_val,
                                    source_type=SourceType.PDF,
                                    location={"element": "metadata", "key": meta_key},
                                    is_hidden=False,
                                    segment_type="PDF_METADATA",
                                    cloaking_signal="METADATA_SUSPICIOUS_CONTENT",
                                    metadata_anomaly_score=40.0,
                                    threat_indicators=[ThreatCategory.METADATA_INJECTION],
                                    confidence_penalty=0.0,
                                    metadata={
                                        "meta_key": meta_key,
                                        "anomaly_codes": ["METADATA_SUSPICIOUS_CONTENT"],
                                        "anomaly_code": "METADATA_SUSPICIOUS_CONTENT"
                                    },
                                )
                            )

            # ----------------------------------------------------
            # 2. Pages, Structured Blocks, and Anomaly Separation
            # ----------------------------------------------------
            seen_texts = set()

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_rect = page.rect
                # Extended clip allows capturing off-canvas coordinates outside visible viewport
                extended_clip = fitz.Rect(page_rect.x0 - 1000, page_rect.y0 - 1000, page_rect.x1 + 1500, page_rect.y1 + 1500)
                text_page = page.get_text("dict", clip=extended_clip, flags=fitz.TEXTFLAGS_SEARCH)

                page_anom_spans: List[dict] = []

                for block in text_page.get("blocks", []):
                    if block.get("type") == 0:  # 0 = Text block
                        normal_lines: List[str] = []
                        block_bbox = block.get("bbox", [0, 0, 0, 0])

                        for line in block.get("lines", []):
                            line_normal_spans: List[str] = []

                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if not text:
                                    continue

                                font_size = span.get("size", 10.0)
                                color = span.get("color", 0)  # RGB int
                                bbox = span.get("bbox", [0, 0, 0, 0])

                                is_micro = font_size < self.min_visible_font_size
                                is_white = color in (16777215, 0xFFFFFF)
                                is_offcanvas = (
                                    bbox[0] < -2
                                    or bbox[1] < -2
                                    or bbox[2] > page_rect.width + 5
                                    or bbox[3] > page_rect.height + 5
                                )

                                if is_micro or is_white or is_offcanvas:
                                    page_anom_spans.append({
                                        "text": text,
                                        "font_size": font_size,
                                        "color": color,
                                        "bbox": bbox,
                                        "is_micro": is_micro,
                                        "is_white": is_white,
                                        "is_offcanvas": is_offcanvas,
                                        "font": span.get("font", "Unknown"),
                                    })
                                else:
                                    line_normal_spans.append(text)

                            if line_normal_spans:
                                normal_lines.append(" ".join(line_normal_spans))

                        if normal_lines:
                            full_block_text = " ".join(normal_lines).strip()
                            if full_block_text and full_block_text not in seen_texts:
                                seen_texts.add(full_block_text)
                                result.segments.append(
                                    ExtractedSegment(
                                        content=full_block_text,
                                        source_type=SourceType.PDF,
                                        location={"page": page_num + 1, "bbox": [round(x, 1) for x in block_bbox]},
                                        is_hidden=False,
                                        segment_type="DOCUMENT_TEXT",
                                        cloaking_signal="NONE",
                                        layout_anomaly_score=0.0,
                                        visibility_score=0.0,
                                        position_anomaly_score=0.0,
                                        metadata_anomaly_score=0.0,
                                        threat_indicators=[],
                                        confidence_penalty=0.0,
                                        metadata={
                                            "page": page_num + 1,
                                            "bbox": [round(x, 1) for x in block_bbox],
                                        },
                                    )
                                )

                # Aggregate anomalous spans at the line baseline level (Phase 1, Item 4)
                if page_anom_spans:
                    # Sort top to bottom, then left to right
                    page_anom_spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
                    
                    line_clusters: List[List[dict]] = []
                    for sp in page_anom_spans:
                        if not line_clusters:
                            line_clusters.append([sp])
                        else:
                            last_cluster = line_clusters[-1]
                            avg_y = sum(s["bbox"][1] for s in last_cluster) / len(last_cluster)
                            if abs(sp["bbox"][1] - avg_y) <= 3.0:
                                last_cluster.append(sp)
                            else:
                                line_clusters.append([sp])

                    for cluster in line_clusters:
                        cluster.sort(key=lambda s: s["bbox"][0])
                        line_anom_text = " ".join(s["text"] for s in cluster).strip()
                        if not line_anom_text or line_anom_text in seen_texts:
                            continue
                        seen_texts.add(line_anom_text)

                        signals = []
                        layout_score = 0.0
                        vis_score = 0.0
                        pos_score = 0.0
                        min_fs = min(s["font_size"] for s in cluster)
                        has_micro = any(s["is_micro"] for s in cluster)
                        has_white = any(s["is_white"] for s in cluster)
                        has_offcanvas = any(s["is_offcanvas"] for s in cluster)

                        if has_micro:
                            signals.append("LAYOUT_ANOMALY_SMALL_FONT")
                            layout_score = 45.0 if min_fs <= 1.5 else 30.0
                        if has_white:
                            signals.append("VISIBILITY_ANOMALY_WHITE_TEXT")
                            vis_score = 50.0
                        if has_offcanvas:
                            signals.append("POSITION_ANOMALY_OFF_CANVAS")
                            pos_score = 40.0

                        min_x0 = min(s["bbox"][0] for s in cluster)
                        min_y0 = min(s["bbox"][1] for s in cluster)
                        max_x1 = max(s["bbox"][2] for s in cluster)
                        max_y1 = max(s["bbox"][3] for s in cluster)
                        anom_bbox = [round(min_x0, 1), round(min_y0, 1), round(max_x1, 1), round(max_y1, 1)]

                        cloaking_sig = "MICRO_FONT" if (has_micro and not has_white and not has_offcanvas) else ("WHITE_TEXT" if has_white else ("OFF_CANVAS" if has_offcanvas else "_".join(signals)))

                        result.segments.append(
                            ExtractedSegment(
                                content=line_anom_text,
                                source_type=SourceType.PDF,
                                location={"page": page_num + 1, "bbox": anom_bbox},
                                is_hidden=True,
                                segment_type="CLOAKED_TEXT",
                                cloaking_signal=cloaking_sig,
                                layout_anomaly_score=layout_score,
                                visibility_score=vis_score,
                                position_anomaly_score=pos_score,
                                threat_indicators=[ThreatCategory.HIDDEN_TEXT],
                                confidence_penalty=0.0,
                                metadata={
                                    "page": page_num + 1,
                                    "font_size": round(min_fs, 2),
                                    "font_name": cluster[0]["font"],
                                    "color_hex": hex(cluster[0]["color"]),
                                    "bbox": anom_bbox,
                                    "anomaly_codes": signals,
                                },
                            )
                        )

                # ----------------------------------------------------
                # 3. PDF Annotations & Sticky Notes Inspection
                # ----------------------------------------------------
                try:
                    annots = page.annots()
                    if annots:
                        for annot in annots:
                            info = annot.info
                            content = info.get("content", "").strip() if info else ""
                            if content:
                                has_annot_attack = bool(SUSPICIOUS_METADATA_REGEX.search(content))
                                annot_anom_codes = ["ANNOTATION_SUSPICIOUS_CONTENT"] if has_annot_attack else []
                                result.segments.append(
                                    ExtractedSegment(
                                        content=content,
                                        source_type=SourceType.PDF,
                                        location={"page": page_num + 1, "element": "annotation", "title": info.get('title', 'Note')},
                                        is_hidden=False,
                                        segment_type="ANNOTATION",
                                        cloaking_signal="ANNOTATION_SUSPICIOUS_CONTENT" if has_annot_attack else "NONE",
                                        layout_anomaly_score=30.0 if has_annot_attack else 0.0,
                                        threat_indicators=[ThreatCategory.METADATA_INJECTION] if has_annot_attack else [],
                                        confidence_penalty=0.0,
                                        metadata={
                                            "annot_type": getattr(annot, "type", ["Unknown"])[1],
                                            "anomaly_codes": annot_anom_codes,
                                        },
                                    )
                                )
                except Exception:
                    pass

            result.raw_character_count = sum(len(s.content) for s in result.segments)
            result.anomalies_detected = sum(
                1 for s in result.segments if s.is_hidden or s.cloaking_signal != "NONE"
            )

        except Exception as e:
            result.extraction_warnings.append(f"PDF extraction error: {str(e)}")
        finally:
            if doc:
                doc.close()

        return result