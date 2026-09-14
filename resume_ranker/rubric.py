"""Deterministic, explainable scoring against a job description.

This is the workhorse. The LLM is good at *judgement* ("is this person
actually a quant?") but bad at *recall* -- a 7B model reading 4 chunks will
miss that the candidate mentioned GARCH once on page 2. Keyword matching
never misses. So we compute a transparent keyword score first, then let the
LLM adjust it. Every hit carries an evidence snippet, which is what ends up
in the CSV's reasoning column.

Matching uses "squashed" text (letters+digits only, lowercased), which makes
it immune to OCR spacing damage and to punctuation variants:
    "Power BI" == "PowerBI" == "Power-BI" == "Power Bl" (bad OCR)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .textutils import squash, squash_with_map, snippet

# --------------------------------------------------------------------------
# Education / experience heuristics
# --------------------------------------------------------------------------
# The leading letter is optional in the Bachelor/Master patterns: OCR drops
# the first character of a word far more often than any other position, so
# real scans regularly contain "achelor of Science" / "aster of Finance".
DEGREE_PATTERNS: list[tuple[str, int, re.Pattern]] = [
    ("PhD", 4, re.compile(r"\b(ph\.?\s?d|doctor(ate|al)|d\.phil)\b", re.I)),
    ("Masters", 3, re.compile(r"\b(m\.?\s?(sc|tech|eng|a|com|ba|fin|s\.c)|m?aster'?s?\b|"
                              r"mba|msc|mtech|meng|mca|mstat|mfin|mfe)\b", re.I)),
    ("Bachelors", 2, re.compile(r"\b(b\.?\s?(sc|tech|eng|a|com|ba|s\.c)|b?achelor'?s?\b|"
                                r"bsc|btech|beng|bca|bba|bcom|undergraduate)\b", re.I)),
    ("Diploma", 1, re.compile(r"\b(diploma|associate'?s?\s+degree|higher\s+secondary|"
                              r"hsc|12th|intermediate)\b", re.I)),
]

FIELD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Quant Finance", re.compile(r"quant(itativ)?e?\s+fin|financial\s+engineer|computational\s+finance", re.I)),
    ("Finance", re.compile(r"\bfinanc(e|ial)\b", re.I)),
    ("Economics", re.compile(r"\beconomic(s|ist)?\b", re.I)),
    ("Statistics", re.compile(r"\bstatistic(s|al)\b|\bbiostatistic", re.I)),
    ("Mathematics", re.compile(r"\bmath(ematics|ematical|s)?\b|\bapplied\s+math", re.I)),
    ("Computer Science", re.compile(r"computer\s+science|\bcs\b|\bsoftware\s+eng|\bcomput(er|ational)\b", re.I)),
    ("Data/Analytics", re.compile(r"data\s+science|analytics|\bml\b|machine\s+learning", re.I)),
    ("Engineering", re.compile(r"\bengineer(ing)?\b", re.I)),
    ("Business", re.compile(r"\bbusiness\s+(admin|management)|\bbba\b|commerce", re.I)),
]

YEAR_RANGE = re.compile(r"(19|20)\d{2}\s*(?:-|–|—|to|until)\s*((19|20)\d{2}|present|current|now|date)", re.I)
EXPLICIT_YEARS = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:professional\s+|relevant\s+|industry\s+|"
    r"hands[\s\-]?on\s+)?(?:experience|exp\b|work)", re.I
)
EXPLICIT_YEARS_ALT = re.compile(r"(?:over|more\s+than|nearly|about|~)?\s*(\d{1,2})\s*\+?\s*years?", re.I)

# Words that mark a year range as academic rather than employment.
# NOTE: "universit" is deliberately absent -- universities are also employers
# ("XYZ University 2017-2022" can be a job). The window is short (60 chars) so
# only text immediately preceding the range is considered.
EDU_CONTEXT_RE = re.compile(
    r"(college|bachelor|master|b\.?\s?s\b|b\.?\s?a\b|mba|gpa|"
    r"degree|school|institut|academ|diploma|graduat|coursework|major)",
    re.I,
)


@dataclass
class SkillHit:
    skill: str
    matched_alias: str
    evidence: str


@dataclass
class CategoryResult:
    name: str
    weight: float
    must_have: bool
    matched: list[SkillHit]
    missing: list[str]
    raw_ratio: float
    points: float
    max_points: float


@dataclass
class RubricResult:
    score: float
    categories: list[CategoryResult]
    matched_skills: list[str]
    missing_skills: list[str]
    must_have_met: list[str]
    must_have_missing: list[str]
    education_level: str
    education_fields: list[str]
    field_relevant: bool
    years_experience: float
    notes: list[str]


class Rubric:
    """A weighted skill rubric loaded from JSON."""

    # Aliases shorter than this many squashed characters are matched on word
    # boundaries in the RAW text instead of as substrings of squashed text.
    #
    # 5 is the sweet spot. Short squashed aliases are a false-positive
    # factory: "r" matches every English sentence (the R language), "cte"
    # matches inside "impacted", "arch" matches "architecture"/"search",
    # "dash" matches "dashboard", "var" matches "variable". Long strings
    # (>=5) essentially never occur inside an unrelated word, so they keep
    # the OCR-robust substring behaviour.
    min_substring_len: int = 5

    def __init__(self, data: dict[str, Any]):
        self.title: str = data.get("title", "Untitled role")
        self.categories: list[dict[str, Any]] = data.get("categories", [])
        self.education_fields: list[str] = [
            f.lower() for f in data.get("education_relevant_fields", [])
        ]
        self.experience_target_years: float = float(data.get("experience_target_years", 3))
        self.synonyms: dict[str, list[str]] = data.get("synonyms", {})
        # Pre-squash every alias once.
        self._prepared: list[tuple[str, float, bool, list[tuple[str, str]]]] = []
        for cat in self.categories:
            aliases: list[tuple[str, str, bool, str]] = []
            for skill in cat.get("skills", []):
                if isinstance(skill, str):
                    skill = {"name": skill, "aliases": [skill]}
                name = skill["name"]
                alts = list(skill.get("aliases") or []) + [name]
                for extra in self.synonyms.get(name, []):
                    alts.append(extra)
                for a in alts:
                    sq = squash(a)
                    if not sq:
                        continue
                    # Short squashed aliases (<3 chars) are dangerous as
                    # substrings: "R" squashes to "r", which appears in
                    # virtually every English sentence -- matching it would
                    # credit every candidate with R programming and satisfy a
                    # must-have. Those are matched on word boundaries instead.
                    is_short = len(sq) < self.min_substring_len
                    aliases.append((name, sq, is_short, a))
            self._prepared.append(
                (cat["name"], float(cat.get("weight", 1.0)), bool(cat.get("must_have", False)), aliases)
            )

    # -- helpers -----------------------------------------------------------
    def all_keywords(self) -> list[str]:
        out: list[str] = []
        for _, _, _, aliases in self._prepared:
            # Substring-usable keywords only (chunk ranking uses squashed text).
            out.extend(sq for _, sq, short, _ in aliases if not short and len(sq) >= 3)
        return sorted(set(out), key=len, reverse=True)

    def _find(self, squashed: str, alias: str) -> tuple[int, int]:
        return (-1, -1) if not alias else ((squashed.find(alias), len(alias)) if alias in squashed else (-1, -1))

    # -- main --------------------------------------------------------------
    def score(self, text: str) -> RubricResult:
        raw = text or ""
        squashed, idx_map = squash_with_map(raw)
        cats: list[CategoryResult] = []
        total_points = 0.0
        all_matched: list[str] = []
        all_missing: list[str] = []
        must_met: list[str] = []
        must_missing: list[str] = []

        for name, weight, must_have, aliases in self._prepared:
            hit_names: set[str] = set()
            hits: list[SkillHit] = []
            groups: dict[str, list[tuple[str, bool, str]]] = {}
            for skill_name, sq, is_short, orig in aliases:
                groups.setdefault(skill_name, []).append((sq, is_short, orig))

            for skill_name, variants in groups.items():
                best = None
                for sq, is_short, orig in variants:
                    if is_short:
                        # Word-boundary match in the RAW text: "R" must appear
                        # as a standalone token, not as any letter 'r'.
                        # A plural/possessive suffix is allowed so "CTEs",
                        # "APIs" and "KPIs" still match.
                        m = re.search(
                            r"(?<![A-Za-z0-9])" + re.escape(orig) + r"(?:s|es)?(?![A-Za-z0-9])",
                            raw, re.I,
                        )
                        if m and (best is None or len(orig) > len(best[0])):
                            best = (orig, snippet(raw, m.start(), m.end()), len(orig))
                        continue
                    pos = squashed.find(sq)
                    if pos >= 0:
                        # Prefer the longest alias that matched (most specific).
                        if best is None or len(sq) > len(best[0]):
                            # Map the squashed span back into raw indices so the
                            # evidence snippet comes from the right sentence.
                            start = idx_map[pos]
                            end = idx_map[min(pos + len(sq) - 1, len(idx_map) - 1)] + 1
                            best = (sq, snippet(raw, start, end), len(sq))
                if best:
                    alias, evidence, _ = best
                    hit_names.add(skill_name)
                    hits.append(
                        SkillHit(skill=skill_name, matched_alias=alias, evidence=evidence)
                    )

            total_groups = len(groups) or 1
            ratio = len(hit_names) / total_groups
            # Diminishing returns: 1 of 8 skills in a category is worth a lot,
            # the 8th adds little. exp curve keeps scores spread out nicely.
            pct = 1.0 - math.exp(-3.0 * ratio)
            points = weight * pct
            total_points += points

            missing = [s for s in groups if s not in hit_names]
            cats.append(
                CategoryResult(
                    name=name,
                    weight=weight,
                    must_have=must_have,
                    matched=sorted(hits, key=lambda h: h.skill),
                    missing=sorted(missing),
                    raw_ratio=round(ratio, 4),
                    points=round(points, 3),
                    max_points=weight,
                )
            )
            all_matched.extend(sorted(hit_names))
            all_missing.extend(missing)
            if must_have:
                if hit_names:
                    must_met.append(name)
                else:
                    must_missing.append(name)

        max_total = sum(c[1] for c in self._prepared) or 1.0
        base = 100.0 * total_points / max_total

        edu_level, edu_fields, field_ok = self._education(raw)
        years = self._years(raw)

        return RubricResult(
            score=round(base, 2),
            categories=cats,
            matched_skills=sorted(set(all_matched)),
            missing_skills=sorted(set(all_missing)),
            must_have_met=must_met,
            must_have_missing=must_missing,
            education_level=edu_level,
            education_fields=edu_fields,
            field_relevant=field_ok,
            years_experience=years,
            notes=[],
        )

    # -- education / experience -------------------------------------------
    def _education(self, text: str) -> tuple[str, list[str], bool]:
        level, rank = "", 0
        for name, r, pat in DEGREE_PATTERNS:
            if pat.search(text):
                if r > rank:
                    level, rank = name, r
        if not level:
            # OCR frequently glues the degree onto a neighbouring word
            # ("ScienceBachelor"), which defeats \b. Fall back to a squashed
            # substring test using only long, unambiguous tokens.
            sq = squash(text)
            for token, name, r in (
                ("bachelor", "Bachelors", 2), ("achelor", "Bachelors", 2),
                ("masters", "Masters", 3), ("mastersdegree", "Masters", 3),
                ("masterof", "Masters", 3), ("doctorate", "PhD", 4),
            ):
                if token in sq and r > rank:
                    level, rank = name, r
        fields = [f for f, pat in FIELD_PATTERNS if pat.search(text)]
        field_ok = any(f.lower() in self.education_fields for f in fields)
        return (level or "Unknown"), fields[:4], field_ok

    @staticmethod
    def _work_text(text: str) -> str:
        """Text with education/certification sections removed.

        Date ranges are scanned for career length, and a degree's "2014-2018"
        is not work experience. If the document has no recognisable sections
        (very common after OCR), we fall back to the whole text rather than
        throwing the signal away.
        """
        try:
            from .chunk import split_into_sections

            sections = split_into_sections(text)
        except Exception:
            return text
        names = {n for n, _ in sections}
        if not (names & {"education", "certifications", "publications", "awards"}):
            return text
        keep = "\n".join(
            body for n, body in sections
            if n not in {"education", "certifications", "publications", "awards", "interests"}
        )
        return keep or text

    def _years(self, text: str) -> float:
        work = self._work_text(text)
        best = 0.0
        m = EXPLICIT_YEARS.search(work)
        if m:
            try:
                best = max(best, float(m.group(1)))
            except ValueError:
                pass
        if best == 0.0:
            m2 = EXPLICIT_YEARS_ALT.search(work)
            if m2:
                try:
                    v = float(m2.group(1))
                    if 0 < v <= 45:
                        best = max(best, v)
                except ValueError:
                    pass
        spans: list[tuple[int, int]] = []
        this_year = datetime.now().year
        for m in YEAR_RANGE.finditer(work):
            # Section splitting can't catch everything: in two-column resumes
            # the degree often sits in the header ("BS ... University 2006-2010")
            # before any EDUCATION heading exists. Reject ranges whose
            # immediate context is academic.
            if EDU_CONTEXT_RE.search(work[max(0, m.start() - 60): m.start()]):
                continue
            a = int(re.match(r"((?:19|20)\d{2})", m.group(0)).group(1))
            tail = m.group(2)
            # Never hardcode the end year -- "to Present" must track the clock.
            b = this_year if re.match(r"present|current|now|date", tail, re.I) else int(tail)
            if 1950 <= a <= b <= this_year + 1:
                spans.append((a, b))
        if spans:
            merged = _merge_spans(spans)
            best = max(best, sum(b - a for a, b in merged))
        return round(min(best, 45.0), 1)


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans = sorted(spans)
    out: list[tuple[int, int]] = []
    for a, b in spans:
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def load_rubric(path: str | Path) -> Rubric:
    return Rubric(json.loads(Path(path).read_text(encoding="utf-8")))
