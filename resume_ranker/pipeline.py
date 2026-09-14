"""Orchestration: PDFs -> parsed candidates -> ranked results.

Stages
  1. Extract  (CPU/IO bound : process pool, cached)
  2. Parse    (cheap, deterministic: contacts, sections, keyword rubric)
  3. Judge    (network bound : async, semaphore-limited calls to Ollama)
  4. Blend + rank, then report

Stage 2 runs while stage 3 is in flight for earlier documents, so the LLM is
never idle waiting on a CPU parse.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import prompts
from .chunk import ChunkConfig, chunk_document, rank_chunks
from .config import Config
from .contacts import parse_contacts
from .extract import ExtractedDoc, extract_all
from .llm import OllamaLLM
from .rubric import Rubric, RubricResult
from .score import JudgeResult, calibrate_deterministic, final_score, parse_judge, recommendation_for

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    file: str
    path: str
    pages: int
    ocr_used: bool
    cached: bool
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    location: str = ""

    det_raw: float = 0.0
    det_calibrated: float = 0.0
    llm_score: float | None = None
    final_score: float = 0.0
    rank: int = 0
    recommendation: str = ""
    blend: str = ""

    years_experience: float = 0.0
    education_level: str = ""
    education_fields: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    must_have_met: list[str] = field(default_factory=list)
    must_have_missing: list[str] = field(default_factory=list)
    category_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    reasoning: str = ""
    llm_reasoning: str = ""
    seniority: str = ""
    llm_confidence: float = 0.0

    parse_status: str = "ok"
    warnings: list[str] = field(default_factory=list)
    llm_calls: int = 0
    llm_seconds: float = 0.0
    n_chunks: int = 0
    chunks_sent: int = 0
    word_count: int = 0


def _build_evidence(det: RubricResult, limit: int = 8) -> list[str]:
    out = []
    for cat in sorted(det.categories, key=lambda c: -c.points):
        for hit in cat.matched[:3]:
            out.append(f"{cat.name}: {hit.skill} — “{hit.evidence}”")
            if len(out) >= limit:
                return out
    return out


def parse_deterministic(doc: ExtractedDoc, rubric: Rubric,
                        chunk_cfg: ChunkConfig) -> tuple[Candidate, list, list]:
    """Stage 2: everything that needs no LLM."""
    cand = Candidate(
        file=doc.file, path=doc.path, pages=doc.pages,
        ocr_used=doc.ocr_used, cached=doc.cached,
        word_count=doc.word_count,
    )

    if doc.error:
        cand.parse_status = "error"
        cand.warnings.append(f"extract:{doc.error}")
    if not doc.text.strip():
        cand.parse_status = "empty"
        cand.warnings.append("no-text-extracted")
        return cand, [], []

    info = parse_contacts(doc.text, filename=doc.file)
    cand.name = info.name
    cand.email = info.email
    cand.phone = info.phone
    cand.linkedin = info.linkedin
    cand.github = info.github
    cand.location = info.location
    cand.warnings.extend(info.warnings)
    if info.email_is_placeholder:
        cand.warnings.append("placeholder-email(template)")

    det = rubric.score(doc.text)
    cand.det_raw = det.score
    cand.years_experience = det.years_experience
    cand.education_level = det.education_level
    cand.education_fields = det.education_fields
    cand.matched_skills = det.matched_skills
    cand.missing_skills = det.missing_skills
    cand.must_have_met = det.must_have_met
    cand.must_have_missing = det.must_have_missing
    cand.category_scores = {c.name: c.points for c in det.categories}
    cand.evidence = _build_evidence(det)

    ch = chunk_document(doc.text, chunk_cfg)
    keep = rank_chunks(ch, rubric.all_keywords(), chunk_cfg)
    cand.n_chunks = len(ch.chunks)
    cand.chunks_sent = len(keep)
    if ch.truncated:
        cand.warnings.append(f"long-resume:{len(ch.chunks)}chunks->{len(keep)}sent")
    return cand, keep, [c.text for c in keep]


async def _judge_one(llm: OllamaLLM, rubric: Rubric, jd_text: str,
                     chunks: list[str], rubric_result: RubricResult,
                     cfg: Config) -> JudgeResult:
    t0 = time.perf_counter()
    joined = "\n\n".join(chunks)
    res = JudgeResult()

    try:
        if len(joined) <= cfg.chunk.single_pass_max_chars:
            data = await llm.chat_json(
                prompt=prompts.judge_prompt(
                    jd_text, rubric, joined,
                    rubric_result.matched_skills, rubric_result.missing_skills,
                ),
                system=prompts.SYSTEM,
                num_ctx=cfg.ollama.reduce_ctx,
                num_predict=cfg.ollama.reduce_predict,
            )
            res = parse_judge(data)
            res.calls = 1
        else:
            # Map over chunks, then reduce. Keeps each call inside a small ctx.
            evidence: list[str] = []
            skills: list[str] = []
            for i, ch in enumerate(chunks):
                d = await llm.chat_json(
                    prompt=prompts.map_prompt(rubric, ch, f"part {i + 1}/{len(chunks)}"),
                    system=prompts.SYSTEM,
                    num_ctx=cfg.ollama.chunk_ctx,
                    num_predict=cfg.ollama.chunk_predict,
                )
                res.calls += 1
                if isinstance(d, dict):
                    for e in d.get("evidence", []) or []:
                        if str(e).strip():
                            evidence.append(str(e)[:200])
                    for s in d.get("skills", []) or []:
                        if str(s).strip():
                            skills.append(str(s)[:60])
            data = await llm.chat_json(
                prompt=prompts.reduce_prompt(
                    jd_text, rubric, evidence, skills,
                    rubric_result.matched_skills, rubric_result.missing_skills,
                ),
                system=prompts.SYSTEM,
                num_ctx=cfg.ollama.reduce_ctx,
                num_predict=cfg.ollama.reduce_predict,
            )
            res = parse_judge(data)
            res.calls += 1
    except Exception as exc:
        res.error = f"{type(exc).__name__}: {exc}"
        log.warning("LLM judge failed: %s", res.error)

    res.seconds = time.perf_counter() - t0
    return res


async def run(paths: list[Path], rubric: Rubric, jd_text: str, cfg: Config,
              progress=None) -> list[Candidate]:
    t_start = time.perf_counter()

    # ---- Stage 1: extraction (blocking, CPU bound) -----------------------
    docs: list[ExtractedDoc] = await asyncio.to_thread(extract_all, paths, cfg.extract)
    ok = [d for d in docs if d.text.strip()]
    log.info("Extracted %d/%d PDFs with text (%d from cache, %d needed OCR).",
             len(ok), len(docs), sum(1 for d in docs if d.cached),
             sum(1 for d in docs if d.ocr_used))

    # ---- Stage 2: deterministic parse ------------------------------------
    cands: list[Candidate] = []
    chunk_lists: list[list[str]] = []
    det_results: list[RubricResult] = []
    for d in docs:
        cand, keep, texts = parse_deterministic(d, rubric, cfg.chunk)
        cands.append(cand)
        chunk_lists.append(texts)
        det_results.append(rubric.score(d.text) if d.text.strip() else RubricResult(
            score=0.0, categories=[], matched_skills=[], missing_skills=[],
            must_have_met=[], must_have_missing=[], education_level="Unknown",
            education_fields=[], field_relevant=False, years_experience=0.0, notes=[]))
        if progress:
            progress(f"parsed {cand.file}")

    # ---- Stage 3: LLM judging --------------------------------------------
    judges: list[JudgeResult | None] = [None] * len(cands)
    if cfg.use_llm:
        async with OllamaLLM(cfg.ollama) as llm:
            try:
                health = await llm.health()
            except Exception as exc:
                log.error("Ollama unreachable (%s). Falling back to keyword-only "
                          "scoring. Start the server with `ollama serve`.", exc)
                health = None

            if health is not None:
                if not health["model_present"]:
                    raise SystemExit(
                        f"Model '{cfg.ollama.model}' is not pulled on the Ollama server.\n"
                        f"Run:  ollama pull {cfg.ollama.model}\n"
                        f"Available: {', '.join(health['models']) or 'none'}"
                    )
                await llm.warmup()

                async def judge(i: int) -> None:
                    if not chunk_lists[i] or cands[i].parse_status == "error":
                        return
                    judges[i] = await _judge_one(llm, rubric, jd_text, chunk_lists[i],
                                                 det_results[i], cfg)
                    if progress:
                        progress(f"judged {cands[i].file}")

                await asyncio.gather(*(judge(i) for i in range(len(cands))))

    # ---- Stage 4: blend + rank -------------------------------------------
    for cand, det, judge in zip(cands, det_results, judges):
        det_cal, notes = calibrate_deterministic(det, cfg.score)
        cand.det_calibrated = round(det_cal, 2)
        if judge is not None:
            cand.llm_score = None if judge.score is None else round(judge.score, 2)
            cand.llm_reasoning = judge.reasoning
            cand.strengths = judge.strengths
            cand.gaps = judge.gaps
            cand.seniority = judge.seniority
            cand.llm_confidence = judge.confidence
            cand.llm_calls = judge.calls
            cand.llm_seconds = round(judge.seconds, 2)
            # Years of experience: prefer our own evidence-based estimate
            # (explicit "X years of experience" statements + employment date
            # ranges). Only fall back to the model when we found nothing.
            # NB: do NOT use max() here -- that lets a confident-but-wrong
            # model inflate every candidate to the same number, and hides
            # disagreement instead of surfacing it.
            if judge.years_relevant is not None:
                det_years = cand.years_experience or 0.0
                llm_years = float(judge.years_relevant)
                if det_years <= 0:
                    cand.years_experience = round(llm_years, 1)
                elif abs(det_years - llm_years) >= 4:
                    cand.warnings.append(
                        f"years-disagreement(det {det_years:g} vs llm {llm_years:g})")
            if judge.error:
                cand.warnings.append(f"llm:{judge.error[:80]}")
                cand.parse_status = "llm-failed"
            elif judge.score is not None and abs(judge.score - det_cal) > cfg.score.disagreement_threshold:
                cand.warnings.append(
                    f"scorer-disagreement(det {det_cal:.0f} vs llm {judge.score:.0f})")

        cand.final_score, cand.blend = final_score(det, det_cal, judge, cfg.score)
        cand.final_score = round(cand.final_score, 2)
        cand.recommendation = recommendation_for(cand.final_score, cfg.score)

        if not cand.reasoning:
            bits = []
            if cand.must_have_met:
                bits.append("Meets: " + ", ".join(cand.must_have_met))
            if cand.must_have_missing:
                bits.append("Missing: " + ", ".join(cand.must_have_missing))
            bits.extend(notes)
            if judge and judge.ok:
                bits.append(judge.reasoning)
            cand.reasoning = " | ".join(b for b in bits if b)

    # Flag byte-identical PDFs so you don't interview the same CV twice.
    seen_hashes: dict[str, Candidate] = {}
    for cand, doc in zip(cands, docs):
        h = getattr(doc, "sha1", "")
        if not h:
            continue
        if h in seen_hashes:
            cand.warnings.append(f"duplicate-of:{seen_hashes[h].file}")
        else:
            seen_hashes[h] = cand

    cands.sort(key=lambda c: (-c.final_score, -c.det_calibrated, c.file))
    for i, c in enumerate(cands, 1):
        c.rank = i

    log.info("Ranked %d candidates in %.1fs.", len(cands), time.perf_counter() - t_start)
    return cands


def to_jsonable(cands: list[Candidate]) -> list[dict]:
    return [
        {k: v for k, v in c.__dict__.items()} for c in cands
    ]


def dump_json(cands: list[Candidate], path: Path) -> None:
    path.write_text(json.dumps(to_jsonable(cands), indent=2, default=str), encoding="utf-8")
