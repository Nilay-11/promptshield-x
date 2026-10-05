"""
PromptShield X - Web & HTML Extractor
Parses raw HTML / web scraping responses and detects CSS-cloaked DOM nodes, hidden comments, and zero-width injection strings.
"""

import re
from bs4 import BeautifulSoup, Comment
from typing import Optional, List
from app.modules.extractors import BaseExtractor, ExtractedSegment, ExtractionResult, SourceType, ThreatCategory


class WebExtractor(BaseExtractor):
    ZERO_WIDTH_REGEX = re.compile(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]")
    
    HIDDEN_CSS_PATTERNS = [
        re.compile(r"display\s*:\s*none", re.IGNORECASE),
        re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
        re.compile(r"opacity\s*:\s*0(\.0+)?(?![0-9])", re.IGNORECASE),
        re.compile(r"font-size\s*:\s*0(px|pt|em|rem)?", re.IGNORECASE),
        re.compile(r"(left|top|margin-left|margin-top)\s*:\s*-[0-9]{4,}px", re.IGNORECASE),
        re.compile(r"z-index\s*:\s*-[0-9]+", re.IGNORECASE),
    ]

    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        result = ExtractionResult(source_type=SourceType.WEB, filename=filename or "webpage.html")
        try:
            html_content = data.decode("utf-8", errors="replace")
            soup = BeautifulSoup(html_content, "html.parser")

            # 1. Extract HTML Comments
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for idx, comm in enumerate(comments):
                comm_str = str(comm).strip()
                if len(comm_str) > 5:
                    result.segments.append(
                        ExtractedSegment(
                            content=comm_str,
                            source_type=SourceType.WEB,
                            location=f"HTML:Comment_{idx+1}",
                            is_hidden=True,
                            threat_indicators=[ThreatCategory.HIDDEN_TEXT, ThreatCategory.DOM_CLOAKING],
                            confidence_penalty=40.0
                        )
                    )

            # 2. Remove script and style tags from DOM tree inspection
            for tag in soup(["script", "style", "noscript", "svg"]):
                tag.decompose()

            # 3. Traverse visible and hidden DOM elements
            for elem in soup.find_all(True):
                # Check for zero-width characters in element attributes
                for attr_name, attr_val in elem.attrs.items():
                    val_str = str(attr_val)
                    if self.ZERO_WIDTH_REGEX.search(val_str):
                        clean_attr = self.ZERO_WIDTH_REGEX.sub("", val_str)
                        result.segments.append(
                            ExtractedSegment(
                                content=clean_attr,
                                source_type=SourceType.WEB,
                                location=f"DOM:{elem.name}[@{attr_name}]",
                                is_hidden=True,
                                threat_indicators=[ThreatCategory.ZERO_WIDTH_OBSCURITY],
                                confidence_penalty=45.0
                            )
                        )

                # Check inline style cloaking
                style_attr = elem.get("style", "")
                is_css_hidden = any(pattern.search(style_attr) for pattern in self.HIDDEN_CSS_PATTERNS)
                is_aria_hidden = elem.get("aria-hidden") == "true" or elem.get("hidden") is not None
                
                # Check direct text of node using modern string=True
                text = elem.find(string=True, recursive=False)
                if text:
                    text_str = text.strip()
                    if not text_str:
                        continue
                    
                    threats = []
                    penalty = 0.0
                    is_hidden = False

                    if is_css_hidden or is_aria_hidden:
                        is_hidden = True
                        threats.append(ThreatCategory.DOM_CLOAKING)
                        penalty += 45.0

                    if self.ZERO_WIDTH_REGEX.search(text_str):
                        threats.append(ThreatCategory.ZERO_WIDTH_OBSCURITY)
                        penalty += 40.0
                        text_str = self.ZERO_WIDTH_REGEX.sub("", text_str)

                    elem_id = elem.get("id") or elem.get("class") or elem.name
                    result.segments.append(
                        ExtractedSegment(
                            content=text_str,
                            source_type=SourceType.WEB,
                            location=f"DOM:<{elem.name} ({elem_id})>",
                            is_hidden=is_hidden,
                            threat_indicators=threats,
                            confidence_penalty=penalty,
                            metadata={"tag": elem.name, "classes": elem.get("class", [])}
                        )
                    )

            result.raw_character_count = sum(len(s.content) for s in result.segments)
            result.anomalies_detected = sum(1 for s in result.segments if s.is_hidden or s.threat_indicators)

        except Exception as e:
            result.extraction_warnings.append(f"Web extraction error: {str(e)}")

        return result