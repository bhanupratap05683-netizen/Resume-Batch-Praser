"""Small text helpers shared by chunking, rubric matching and the LLM layer."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9]+")

# Canonical resume section headings (also catches OCR-ish variants).
SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("summary", re.compile(r"^\s*(professional\s+)?(summary|objective|profile|about( me)?)\b", re.I)),
    ("experience", re.compile(r"^\s*(work\s+|professional\s+|employment\s+|relevant\s+)?"
                              r"(experience|history|employment)\b", re.I)),
    ("education", re.compile(r"^\s*(education|academics?|academic\s+(background|qualifications?))\b", re.I)),
    ("skills", re.compile(r"^\s*(technical\s+|core\s+)?(skills|competencies|technologies|tools?|"
                          r"tech\s+stack|proficiencies)\b", re.I)),
    ("projects", re.compile(r"^\s*(projects?|portfolio|key\s+projects?|personal\s+projects?)\b", re.I)),
    ("certifications", re.compile(r"^\s*(certifications?|licenses?|credentials?|"
                                  r"professional\s+development|training)\b", re.I)),
    ("publications", re.compile(r"^\s*(publications?|research|papers?|patents?)\b", re.I)),
    ("awards", re.compile(r"^\s*(awards?|honors?|achievements?|accomplishments?)\b", re.I)),
    ("interests", re.compile(r"^\s*(interests?|hobbies|activities|extracurricular)\b", re.I)),
]

_HEADING_HINT = re.compile(r"^[A-Z][A-Za-z /&'\-]{2,48}$")


def squash(text: str) -> str:
    """Lowercase and strip everything except letters/digits.

    This is the single most useful trick for OCR'd resumes and for matching
    multi-word skills:  "Power BI", "Power-BI", "PowerBI" and the OCR mess
    "Power Bl" all collapse to "powerbi".
    """
    return _NONALNUM.sub("", (text or "").lower())


def squash_with_map(text: str) -> tuple[str, list[int]]:
    """Squash, but also return `map[i] = index in the ORIGINAL text` of the
    i-th squashed character.

    Needed because evidence snippets must be sliced out of the raw text:
    using a squashed index against raw text drifts by hundreds of characters
    (every space and newline removed shifts everything), which silently
    produces snippets from the wrong part of the resume.
    """
    text = text or ""
    out: list[str] = []
    idx: list[int] = []
    i = 0
    for ch in text:
        low = ch.lower()
        # Must match squash() EXACTLY: keep only [a-z0-9]. Accented and CJK
        # characters (OCR noise like 帕/曾) are dropped. If the two functions
        # disagree, indices drift and keyword matches silently break.
        if ("a" <= low <= "z") or ("0" <= low <= "9"):
            out.append(low)
            idx.append(i)
        i += 1
    return "".join(out), idx


def squash_spaces(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def clean_lines(text: str) -> list[str]:
    out = []
    for ln in (text or "").splitlines():
        s = squash_spaces(ln)
        if s:
            out.append(s)
    return out


def detect_section(line: str) -> str | None:
    s = line.strip().strip(":·-|")
    if not s or len(s) > 60:
        return None
    for name, pat in SECTION_PATTERNS:
        if pat.match(s):
            return name
    # A short ALL-CAPS / Title-Case line with no terminal punctuation is
    # very often a section heading that our patterns missed.
    if _HEADING_HINT.match(s) and not s.endswith((".", ",", ";")):
        letters = [c for c in s if c.isalpha()]
        if letters:
            caps = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps > 0.8 and len(s.split()) <= 5:
                return "other"
    return None


def snippet(text: str, start: int, end: int, pad: int = 45) -> str:
    """A readable evidence snippet around a match (for the CSV 'why')."""
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    frag = text[lo:hi].replace("\n", " ")
    frag = _WS.sub(" ", frag).strip()
    return ("…" if lo > 0 else "") + frag + ("…" if hi < len(text) else "")
