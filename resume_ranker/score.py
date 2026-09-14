"""Blend the deterministic rubric score with the LLM judge score."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import ScoreConfig
from .rubric import RubricResult

log = logging.getLogger(__name__)


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class JudgeResult:
    """What the LLM contributes (all optional -- the pipeline survives without it)."""
    score: float | None = None
    confidence: float = 0.0
    years_relevant: float | None = None
    seniority: str = ""
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    reasoning: str = ""
    recommendation: str = ""
    error: str | None = None
    calls: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.score is not None


def parse_judge(data: Any) -> JudgeResult:
    """Tolerant parsing of whatever the model returned."""
    if not isinstance(data, dict):
        return JudgeResult(error=f"unexpected JSON type: {type(data).__name__}")
    out = JudgeResult()
    try:
        s = data.get("score")
        if isinstance(s, (int, float)):
            out.score = clamp(float(s))
        elif isinstance(s, str) and s.strip().isdigit():
            out.score = clamp(float(s.strip()))
    except Exception:
        pass
    try:
        out.confidence = clamp(float(data.get("confidence", 0) or 0), 0, 1)
    except Exception:
        pass
    try:
        y = data.get("years_relevant")
        out.years_relevant = float(y) if isinstance(y, (int, float)) else None
    except Exception:
        pass
    out.seniority = str(data.get("seniority", "") or "")[:32]
    for key, attr in (("strengths", "strengths"), ("gaps", "gaps")):
        val = data.get(key)
        if isinstance(val, list):
            setattr(out, attr, [str(v)[:220] for v in val if str(v).strip()][:5])
        elif isinstance(val, str) and val.strip():
            setattr(out, attr, [val.strip()[:220]])
    out.reasoning = str(data.get("reasoning", "") or "").strip()[:900]
    rec = str(data.get("recommendation", "") or "").strip().lower()
    if rec in ("strong_fit", "moderate_fit", "weak_fit"):
        out.recommendation = rec
    if out.score is None:
        out.error = "no 'score' field in model output"
    return out


def calibrate_deterministic(det: RubricResult, cfg: ScoreConfig) -> tuple[float, list[str]]:
    """Turn raw keyword coverage into a realistic 0-100 score.

    Keyword coverage alone tops out around 60-70 even for a great resume
    (nobody lists every alias), so we apply a gain, then add an experience
    bonus and subtract penalties for missing hard requirements.
    """
    notes: list[str] = []
    base = det.score * cfg.deterministic_gain

    exp_bonus = min(det.years_experience, cfg.experience_bonus_cap_years) * cfg.experience_bonus_per_year
    if exp_bonus:
        notes.append(f"+{exp_bonus:.1f} for {det.years_experience:g}y experience")

    penalty = 0.0
    if det.must_have_missing:
        penalty += cfg.missing_must_have_penalty * len(det.must_have_missing)
        notes.append(f"-{penalty:.1f} missing must-have: {', '.join(det.must_have_missing)}")
    if det.education_level != "Unknown" and not det.field_relevant:
        penalty += cfg.unrelated_degree_penalty
        notes.append(f"-{cfg.unrelated_degree_penalty:.1f} degree field outside target list")

    return clamp(base + exp_bonus - penalty, 0.0, cfg.deterministic_cap), notes


def final_score(det: RubricResult, det_cal: float, judge: JudgeResult | None,
                cfg: ScoreConfig) -> tuple[float, str]:
    if judge is None or not judge.ok:
        return det_cal, "deterministic-only"

    if cfg.weight_llm <= 0:
        return det_cal, "deterministic-only"
    if cfg.weight_deterministic <= 0:
        return judge.score or 0.0, "llm-only"

    total = cfg.weight_deterministic + cfg.weight_llm
    blended = (cfg.weight_deterministic * det_cal + cfg.weight_llm * (judge.score or 0.0)) / total
    return clamp(blended), f"blend {cfg.weight_deterministic:g}*{det_cal:.1f} + {cfg.weight_llm:g}*{judge.score:.1f}"


def recommendation_for(score: float, cfg: ScoreConfig) -> str:
    if score >= cfg.strong_fit:
        return "strong_fit"
    if score >= cfg.moderate_fit:
        return "moderate_fit"
    return "weak_fit"
