"""
PromptShield X - Multimodal Image Extractor
Extracts text from images using OCR and inspects EXIF/IPTC metadata for covert prompt injection payloads.
"""

import io
import re
from PIL import Image, ExifTags
from typing import Optional
from app.modules.extractors import BaseExtractor, ExtractedSegment, ExtractionResult, SourceType, ThreatCategory

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# preferred: RapidOCR 3 with the English PP-OCRv5 recognizer (keeps word spacing);
# fallback: rapidocr_onnxruntime's default Chinese/English model (drops spaces, so merged words are re-split)
try:
    from rapidocr import RapidOCR as _RapidOCR3, LangRec, OCRVersion, ModelType
    HAS_RAPIDOCR3 = True
except ImportError:
    HAS_RAPIDOCR3 = False
try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPIDOCR_LEGACY = True
except ImportError:
    HAS_RAPIDOCR_LEGACY = False
HAS_RAPIDOCR = HAS_RAPIDOCR3 or HAS_RAPIDOCR_LEGACY

_RAPID = None


def _split_merged_words(text: str, min_len: int = 14) -> str:
    """RapidOCR's recognizer often drops spaces in English ("Youarenowdan"). Split only long merged tokens:
    on 100 scanned PDFs this cut false positives from 10% to 2% (eval/results/scanned_pdf_ocr.json)."""
    try:
        import wordninja
    except ImportError:
        return text
    out = []
    for tok in text.split(" "):
        if len(re.sub(r"[^A-Za-z]", "", tok)) >= min_len:
            parts = wordninja.split(tok)
            tok = " ".join(parts) if len(parts) > 1 else tok
        out.append(tok)
    return " ".join(out)


def _reading_order(items):
    """Group OCR boxes into lines (vertical centres closer than half a box height), then sort left-to-right."""
    def geo(it):
        ys = [float(p[1]) for p in it[0]]; xs = [float(p[0]) for p in it[0]]
        return (min(ys) + max(ys)) / 2, max(max(ys) - min(ys), 1.0), min(xs)
    lines = []
    for it in sorted(items, key=lambda it: geo(it)[0]):
        yc, h, _x = geo(it)
        if lines and abs(yc - lines[-1]["yc"]) < 0.5 * max(h, lines[-1]["h"]):
            lines[-1]["items"].append(it)
        else:
            lines.append({"yc": yc, "h": h, "items": [it]})
    return [it for ln in lines for it in sorted(ln["items"], key=lambda it: geo(it)[2])]


def _rapid_ocr(image) -> str:
    global _RAPID
    import numpy as np
    from PIL import ImageOps
    if _RAPID is None:
        _RAPID = None
        if HAS_RAPIDOCR3:
            try:
                eng = _RapidOCR3(params={"Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
                                         "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.lang_type": LangRec.EN,
                                         "Rec.model_type": ModelType.MOBILE})
                def _run3(arr, _e=eng):
                    r = _e(arr)
                    return _reading_order(list(zip(r.boxes, r.txts, r.scores)) if r.txts else [])
                _RAPID = _run3
            except Exception:
                _RAPID = None
        if _RAPID is None:
            legacy = RapidOCR()
            _RAPID = lambda arr, _e=legacy: (_e(arr)[0] or [])
    lines, seen = [], set()
    # second pass on an auto-contrasted copy recovers faint / low-contrast text often used to hide payloads
    for img in (image.convert("RGB"), ImageOps.autocontrast(image.convert("L"), cutoff=1).convert("RGB")):
        for _box, text, conf in _RAPID(np.array(img)):
            key = "".join(text.lower().split())  # spacing differs between passes
            if float(conf) >= 0.5 and key and key not in seen:
                seen.add(key); lines.append(_split_merged_words(text.strip()))
    return chr(10).join(lines)


class ImageExtractor(BaseExtractor):
    def __init__(self, ocr_engine: str = "pytesseract"):
        self.ocr_engine = ocr_engine

    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        result = ExtractionResult(source_type=SourceType.IMAGE, filename=filename)
        try:
            image = Image.open(io.BytesIO(data))
            
            # 1. EXIF Metadata Extraction & Inspection
            exif_data = image.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, bytes)):
                        text_val = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
                        text_val = text_val.strip()
                        if len(text_val) > 4:
                            result.segments.append(
                                ExtractedSegment(
                                    content=text_val,
                                    source_type=SourceType.IMAGE,
                                    location={"element": "exif", "tag": tag_name},
                                    threat_indicators=[ThreatCategory.METADATA_INJECTION],
                                    confidence_penalty=25.0,
                                    metadata={"exif_tag": tag_name}
                                )
                            )

            # 2. OCR Optical Text Extraction (Visual Prompt Injection)
            if HAS_RAPIDOCR or HAS_TESSERACT:
                ocr_text = (_rapid_ocr(image) if HAS_RAPIDOCR else pytesseract.image_to_string(image)).strip()
                if HAS_RAPIDOCR:  # RapidOCR returns lines; keep them as one block so sentences stay together
                    ocr_text = ocr_text.replace(chr(10), " ")
                if ocr_text:
                    # Parse OCR line by line or by paragraph
                    paragraphs = [p.strip() for p in ocr_text.split("\n\n") if p.strip()]
                    for idx, para in enumerate(paragraphs):
                        result.segments.append(
                            ExtractedSegment(
                                content=para,
                                source_type=SourceType.IMAGE,
                                location={"element": "ocr_block", "block_id": idx+1},
                                is_hidden=False,
                                threat_indicators=[ThreatCategory.VISUAL_PROMPT_INJECTION],
                                metadata={"image_format": image.format, "image_size": image.size}
                            )
                        )
            else:
                result.extraction_warnings.append("No OCR engine installed (rapidocr_onnxruntime or pytesseract); OCR skipped.")

            result.raw_character_count = sum(len(s.content) for s in result.segments)
            result.anomalies_detected = sum(1 for s in result.segments if s.is_hidden or s.threat_indicators)

        except Exception as e:
            result.extraction_warnings.append(f"Image extraction error: {str(e)}")

        return result