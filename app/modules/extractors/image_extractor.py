"""
PromptShield X - Multimodal Image Extractor
Extracts text from images using OCR and inspects EXIF/IPTC metadata for covert prompt injection payloads.
"""

import io
from PIL import Image, ExifTags
from typing import Optional
from app.modules.extractors import BaseExtractor, ExtractedSegment, ExtractionResult, SourceType, ThreatCategory

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


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
            if HAS_TESSERACT:
                ocr_text = pytesseract.image_to_string(image).strip()
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
                result.extraction_warnings.append("pytesseract is not installed; OCR extraction was skipped.")

            result.raw_character_count = sum(len(s.content) for s in result.segments)
            result.anomalies_detected = sum(1 for s in result.segments if s.is_hidden or s.threat_indicators)

        except Exception as e:
            result.extraction_warnings.append(f"Image extraction error: {str(e)}")

        return result