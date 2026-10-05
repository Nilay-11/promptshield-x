"""
Phase 2 Input Normalization Engine for PromptShield X.
Provides systematic de-obfuscation across 8 distinct attack vectors:
1. Unicode NFKC normalization
2. Zero-width and invisible character stripping
3. Cyrillic and Greek homoglyph normalization to Latin
4. Leetspeak decoding (digit/symbol to letter mapping)
5. Spaced-out text collapse ("i g n o r e" -> "ignore")
6. Base64 payload detection and decoding
7. ROT13 payload detection and decoding
8. Permutation & fuzzy matching for canonical attack phrases

All components are strictly governed by configuration flags in app.core.settings.
"""

import base64
import codecs
import re
import unicodedata
from typing import Dict, Any, List, Set, Tuple

from app.core.settings import settings


# ----------------------------------------------------------------------
# 2. Zero-width and Invisible Characters
# ----------------------------------------------------------------------
INVISIBLE_CHARS: List[str] = [
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u2060",  # word joiner
    "\ufeff",  # zero-width no-break space / byte order mark
    "\u00ad",  # soft hyphen
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
]


def strip_zero_width_chars(text: str) -> str:
    """Strips zero-width and directional control characters."""
    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")
    return text


# ----------------------------------------------------------------------
# 1. Unicode NFKC Normalization
# ----------------------------------------------------------------------
def normalize_unicode_nfkc(text: str) -> str:
    """Applies NFKC compatibility decomposition and composition."""
    return unicodedata.normalize("NFKC", text)


# ----------------------------------------------------------------------
# 3. Homoglyph Normalization (Cyrillic & Greek to Latin)
# ----------------------------------------------------------------------
HOMOGLYPH_MAP: Dict[str, str] = {
    # Cyrillic lowercase
    "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u0456": "i", "\u0458": "j",
    "\u043e": "o", "\u0440": "p", "\u0455": "s", "\u0445": "x", "\u0443": "y",
    "\u0501": "d", "\u051b": "q", "\u051d": "w", "\u0475": "v",
    # Cyrillic uppercase
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E", "\u041d": "H",
    "\u0406": "I", "\u0408": "J", "\u041a": "K", "\u041c": "M", "\u041e": "O",
    "\u0420": "P", "\u0405": "S", "\u0422": "T", "\u0425": "X", "\u04ae": "Y",
    "\u051c": "W",
    # Greek lowercase
    "\u03b1": "a", "\u03b2": "b", "\u03b3": "y", "\u03b5": "e", "\u03b9": "i",
    "\u03ba": "k", "\u03bd": "v", "\u03bf": "o", "\u03c1": "p", "\u03c4": "t",
    "\u03c5": "u", "\u03c7": "x",
    # Greek uppercase
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X",
}


def normalize_homoglyphs(text: str) -> str:
    """Normalizes Cyrillic and Greek lookalikes to their Latin equivalents."""
    return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in text)


# ----------------------------------------------------------------------
# 4. Leetspeak Decoding
# ----------------------------------------------------------------------
LEET_MAP: Dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "@": "a",
    "$": "s",
    "5": "s",
    "7": "t",
    "!": "i",
}

CANONICAL_VOCABULARY: Set[str] = {
    "ignore", "previous", "instructions", "instruction", "disregard",
    "reveal", "system", "prompt", "bypass", "jailbreak", "override",
    "password", "secret", "unrestricted", "assistant", "rule", "rules"
}


def decode_leetspeak(text: str) -> str:
    """
    Decodes leetspeak substitutions within words (e.g. 'syst3m', '1gnore', 'p@ssword').
    Leaves legitimate standalone numbers, currency values ($50), or percentages intact.
    """
    def _decode_word(token: str) -> str:
        # Match surrounding non-alphanumeric punctuation (except leet symbols @$!)
        m = re.match(r"^([^a-zA-Z0-9@$!]*)([\w@$!]+)([^a-zA-Z0-9@$!]*)$", token)
        if not m:
            return token
        prefix, core, suffix = m.groups()

        has_letters = bool(re.search(r"[a-zA-Z]", core))
        has_leet = bool(re.search(r"[013457@$!]", core))

        if not has_leet:
            return token

        translated = "".join(LEET_MAP.get(c, c) for c in core)

        # If token was mixed alphanumeric/symbolic (e.g. syst3m, 1gnore, $ystem, p@ssword)
        if has_letters:
            return f"{prefix}{translated}{suffix}"

        # If token was purely digits/symbols, only translate if it resolves to a known keyword
        if translated.lower() in CANONICAL_VOCABULARY:
            return f"{prefix}{translated}{suffix}"

        return token

    # Tokenize preserving whitespace
    parts = re.split(r"(\s+)", text)
    result = []
    for part in parts:
        if part.isspace():
            result.append(part)
        else:
            result.append(_decode_word(part))
    return "".join(result)


# ----------------------------------------------------------------------
# 5. Spaced-out Text Collapse
# ----------------------------------------------------------------------
def collapse_spaced_text(text: str) -> str:
    """
    Collapses sequences of single characters separated by spaces (e.g. 'i g n o r e' -> 'ignore').
    Preserves multi-space boundaries between distinct words and leaves natural prose alone.
    """
    # Split by multi-space delimiters so boundaries between separate spaced words are preserved
    parts = re.split(r"(\s{2,})", text)
    processed = []
    # Pattern matches 3 or more single letters separated strictly by a single space
    single_letter_pattern = r"\b[A-Za-z](?:\s[A-Za-z]){2,}\b"

    for part in parts:
        if re.match(r"^\s{2,}$", part):
            processed.append(" ")
        else:
            collapsed = re.sub(
                single_letter_pattern,
                lambda m: re.sub(r"\s", "", m.group(0)),
                part
            )
            processed.append(collapsed)
    return "".join(processed)


# ----------------------------------------------------------------------
# 6. Base64 Payload Detection & Decoding
# ----------------------------------------------------------------------
def detect_and_decode_base64(text: str) -> Tuple[str, List[str]]:
    """
    Detects base64 encoded strings within text, decodes valid ASCII/UTF-8 payloads,
    and appends the decoded payload to the text for downstream evaluation.
    """
    b64_pattern = r"\b[A-Za-z0-9+/]{12,}={0,2}\b"
    matches = re.findall(b64_pattern, text)
    decoded_payloads = []

    for m in matches:
        try:
            # Must be valid length for padding
            pad_len = len(m) % 4
            candidate = m + ("=" * (4 - pad_len) if pad_len != 0 else "")
            raw_bytes = base64.b64decode(candidate, validate=True)
            decoded_str = raw_bytes.decode("utf-8")
            # Verify decoded string is mostly printable text
            if len(decoded_str.strip()) >= 4 and all(32 <= ord(c) < 127 or c in "\r\n\t" for c in decoded_str):
                decoded_payloads.append(decoded_str.strip())
        except Exception:
            continue

    if decoded_payloads:
        appended = text + " " + " ".join(f"[DECODED_BASE64: {p}]" for p in decoded_payloads)
        return appended, decoded_payloads

    return text, []


# ----------------------------------------------------------------------
# 7. ROT13 Payload Detection & Decoding
# ----------------------------------------------------------------------
def detect_and_decode_rot13(text: str) -> Tuple[str, List[str]]:
    """
    Checks if applying ROT13 reveals canonical prompt injection or extraction keywords.
    If detected, appends the decoded payload for downstream evaluation.
    """
    decoded = codecs.decode(text, "rot_13")
    dl = decoded.lower()

    # Check for adversarial triggers in the ROT13 representation
    triggers = [
        "ignore previous", "disregard all", "reveal the system prompt",
        "system prompt", "instructions", "bypass safety", "jailbreak",
        "override", "pretend no rules"
    ]

    found = [t for t in triggers if t in dl]
    if found:
        appended = f"{text} [DECODED_ROT13: {decoded.strip()}]"
        return appended, [decoded.strip()]

    return text, []


# ----------------------------------------------------------------------
# 8. Permutation & Fuzzy Matching for Canonical Attack Vocabulary
# ----------------------------------------------------------------------
def _levenshtein(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (c1 != c2)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


CANONICAL_TARGET_KEYWORDS: Set[str] = {
    "ignore", "previous", "instructions", "instruction", "disregard",
    "reveal", "system", "prompt", "bypass", "jailbreak", "override"
}


def normalize_fuzzy_canonical(text: str) -> str:
    """
    Detects character transpositions / permutations (e.g. 'ignroe', 'revael', 'porpmt')
    and single-edit typos of canonical attack vocabulary, mapping them back to standard form.
    """
    def _map_token(token: str) -> str:
        m = re.match(r"^([^a-zA-Z]*)([a-zA-Z]+)([^a-zA-Z]*)$", token)
        if not m:
            return token
        prefix, word, suffix = m.groups()
        wl = word.lower()

        # If already exactly a canonical word, leave as is
        if wl in CANONICAL_TARGET_KEYWORDS:
            return token

        # 1. Exact anagram / transposition check (e.g. ignroe -> ignore, revael -> reveal)
        sorted_wl = sorted(wl)
        for canon in CANONICAL_TARGET_KEYWORDS:
            if len(wl) == len(canon) and len(wl) >= 4 and sorted_wl == sorted(canon):
                replacement = canon.upper() if word.isupper() else (canon.capitalize() if word.istitle() else canon)
                return f"{prefix}{replacement}{suffix}"

        # 2. Edit distance <= 1 check for words of length >= 5 (e.g. prevous -> previous, bypas -> bypass)
        if len(wl) >= 5:
            for canon in CANONICAL_TARGET_KEYWORDS:
                if abs(len(wl) - len(canon)) <= 1 and _levenshtein(wl, canon) <= 1:
                    replacement = canon.upper() if word.isupper() else (canon.capitalize() if word.istitle() else canon)
                    return f"{prefix}{replacement}{suffix}"

        return token

    parts = re.split(r"(\s+)", text)
    result = []
    for part in parts:
        if part.isspace():
            result.append(part)
        else:
            result.append(_map_token(part))
    return "".join(result)


# ----------------------------------------------------------------------
# Master Normalization Entrypoint
# ----------------------------------------------------------------------
def normalize_input(text: str) -> str:
    """
    Sequential normalizer executing enabled Phase 2 stages according to settings.
    """
    if not settings.normalizer_enabled or not text:
        return text

    # Stage 2: Zero-width character stripping
    if settings.norm_zero_width:
        text = strip_zero_width_chars(text)

    # Stage 1: Unicode NFKC normalization
    if settings.norm_unicode_nfkc:
        text = normalize_unicode_nfkc(text)

    # Stage 3: Homoglyph normalization (Cyrillic/Greek to Latin)
    if settings.norm_homoglyphs:
        text = normalize_homoglyphs(text)

    # Stage 5: Spaced-out text collapse
    if settings.norm_spaced_text:
        text = collapse_spaced_text(text)

    # Stage 6 & 7: Extract Base64 and ROT13 payloads before leetspeak decoding
    b64_payloads = []
    if settings.norm_base64:
        _, b64_payloads = detect_and_decode_base64(text)

    rot13_payloads = []
    if settings.norm_rot13:
        _, rot13_payloads = detect_and_decode_rot13(text)

    # Stage 4: Leetspeak decoding
    if settings.norm_leetspeak:
        text = decode_leetspeak(text)

    # Stage 8: Permutation & fuzzy matching
    if settings.norm_fuzzy_canonical:
        text = normalize_fuzzy_canonical(text)

    # Append decoded payloads in clean cleartext
    for p in b64_payloads:
        text = f"{text} [DECODED_BASE64: {p}]"
    for r in rot13_payloads:
        text = f"{text} [DECODED_ROT13: {r}]"

    return text
