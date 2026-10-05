"""
PromptShield X - Code & Repository Extractor
Extracts comments, docstrings, and documentation while detecting Trojan Source unicode attacks (CVE-2021-42574)
and assistant-hijacking payloads embedded in source code.
"""

import ast
import re
from typing import Optional, List
from app.modules.extractors import (
    BaseExtractor,
    ExtractedSegment,
    ExtractionResult,
    SourceType,
    ThreatCategory,
)


class CodeExtractor(BaseExtractor):
    # Trojan Source Bidi control characters (RLO, LRO, PDF, etc.)
    BIDI_CONTROL_REGEX = re.compile(r"[\u202A-\u202E\u2066-\u2069]")

    # Adversarial instruction directives often snuck into code comments & docstrings
    DIRECTIVE_INJECTION_REGEX = re.compile(
        r"(\[SYSTEM\]|\[INSTRUCTION\]|\[ASSISTANT\]|developer\s+mode|override\s+safety|"
        r"ignore\s+(all|previous|prior)|reveal\s+(system|prompt|secret)|you\s+are\s+now|"
        r"act\s+as|forget\s+safety|output\s+system\s+prompt)",
        re.IGNORECASE,
    )

    def extract(self, data: bytes, filename: Optional[str] = None, **kwargs) -> ExtractionResult:
        fname = filename or "source.py"
        result = ExtractionResult(source_type=SourceType.CODE, filename=fname)

        try:
            code_text = data.decode("utf-8", errors="replace")

            # ----------------------------------------------------
            # 1. Trojan Source Detection (Unicode Bidirectional Overrides)
            # ----------------------------------------------------
            bidi_matches = list(self.BIDI_CONTROL_REGEX.finditer(code_text))
            if bidi_matches:
                result.segments.append(
                    ExtractedSegment(
                        content=f"[TROJAN_SOURCE_DETECTED] Found {len(bidi_matches)} bidirectional Unicode override characters.",
                        source_type=SourceType.CODE,
                        location=f"{fname}:UnicodeSafety",
                        is_hidden=True,
                        threat_indicators=[ThreatCategory.TROJAN_SOURCE_UNICODE],
                        confidence_penalty=60.0,
                        metadata={"bidi_count": len(bidi_matches)},
                    )
                )

            # ----------------------------------------------------
            # 2. Extract Docstrings and Comments
            # ----------------------------------------------------
            if fname.endswith(".py"):
                self._extract_python_ast(code_text, fname, result)
            else:
                self._extract_generic_comments(code_text, fname, result)

            result.raw_character_count = sum(len(s.content) for s in result.segments)
            result.anomalies_detected = sum(
                1 for s in result.segments if s.is_hidden or len(s.threat_indicators) > 0
            )

        except Exception as e:
            result.extraction_warnings.append(f"Code extraction error: {str(e)}")

        return result

    def _evaluate_comment_text(self, text: str, location: str, result: ExtractionResult):
        """Analyzes comment/docstring text for prompt injection markers and adds a segment."""
        clean_text = text.strip()
        if not clean_text:
            return

        threats: List[ThreatCategory] = []
        penalty = 5.0
        is_hidden = False

        # Check for directive injection keywords in comment/docstring
        if self.DIRECTIVE_INJECTION_REGEX.search(clean_text):
            threats.append(ThreatCategory.COMMENT_HIJACK)
            penalty = 45.0
            is_hidden = True

        result.segments.append(
            ExtractedSegment(
                content=clean_text,
                source_type=SourceType.CODE,
                location=location,
                is_hidden=is_hidden,
                threat_indicators=threats,
                confidence_penalty=penalty,
            )
        )

    def _extract_python_ast(self, code_text: str, filename: str, result: ExtractionResult):
        """Extracts module, class, and function docstrings using Python's AST parser."""
        try:
            tree = ast.parse(code_text, filename=filename)

            # Module-level docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                self._evaluate_comment_text(module_doc, f"{filename}:ModuleDocstring", result)

            # Functions, Async Functions, and Classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node)
                    if doc:
                        self._evaluate_comment_text(
                            doc, f"{filename}:{node.name} (Line {node.lineno})", result
                        )

            # Also extract standalone comments via fallback
            self._extract_generic_comments(code_text, filename, result, skip_docstrings=True)

        except SyntaxError:
            self._extract_generic_comments(code_text, filename, result)

    def _extract_generic_comments(
        self, code_text: str, filename: str, result: ExtractionResult, skip_docstrings: bool = False
    ):
        """Extracts single-line and block comments across various programming languages."""
        lines = code_text.splitlines()
        for idx, line in enumerate(lines, start=1):
            line_str = line.strip()
            if line_str.startswith(("#", "//", "/*", "*", "<!--")):
                self._evaluate_comment_text(line_str, f"{filename}:L{idx}", result)