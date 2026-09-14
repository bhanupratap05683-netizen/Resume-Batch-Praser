"""Prompt templates tuned for small (7B) local models.

Rules that keep a 7B model honest:
  * Short, imperative system prompt; the instruction is repeated *inside* the
    user turn right next to the output contract.
  * Explicit, tiny JSON schema with scalar values -- no nested explosions.
  * "Quote evidence from the resume" forces grounding and kills invention.
  * We hand the model the deterministic pre-scan results, so it corrects our
    keyword matcher instead of starting from zero.
"""

SYSTEM = (
    "You are an expert technical recruiter specialising in quantitative finance "
    "and data analytics roles. You evaluate a candidate resume against a job "
    "description. You are strict, evidence-based, and you never invent facts "
    "that are not present in the resume text. You always reply with a single "
    "valid JSON object and nothing else."
)


def _weights_block(rubric) -> str:
    lines = []
    for name, weight, must, aliases in rubric._prepared:
        skills = sorted({alias[0] for alias in aliases})
        shown = ", ".join(skills[:14])
        extra = f" (+{len(skills) - 14} more)" if len(skills) > 14 else ""
        tag = " [MUST-HAVE]" if must else ""
        lines.append(f"- {name} (weight {weight:g}{tag}): {shown}{extra}")
    return "\n".join(lines)


def judge_prompt(jd_text: str, rubric, resume_text: str,
                 matched: list[str], missing: list[str]) -> str:
    matched_s = ", ".join(matched[:40]) or "none"
    missing_s = ", ".join(missing[:25]) or "none"
    return f"""SCORING RUBRIC (weights sum to 100):
{_weights_block(rubric)}

JOB DESCRIPTION:
\"\"\"{jd_text[:3500]}\"\"\"

CANDIDATE RESUME TEXT (may contain minor OCR errors such as missing spaces):
\"\"\"{resume_text}\"\"\"

Automated keyword pre-scan of this resume:
- Skills found: {matched_s}
- Skills NOT found: {missing_s}

TASK:
Score this candidate from 0 to 100 for the role described above.
Guidance: 90+ = exceptional and directly on-target; 75-89 = strong fit;
58-74 = plausible but gaps; 40-57 = weak overlap; below 40 = wrong profile.
Judge depth of experience, not just keyword presence. Penalise resumes whose
experience is unrelated to finance/analytics even if tools match.

OUTPUT a single JSON object, exactly this shape (no prose, no markdown):
{{"score": <integer 0-100>,
  "confidence": <float 0.0-1.0>,
  "years_relevant": <number, years of relevant experience>,
  "seniority": "<intern|junior|mid|senior|lead>",
  "strengths": ["<max 3 items; each must quote or paraphrase real resume content>"],
  "gaps": ["<max 3 items; what the role needs that is missing or thin>"],
  "reasoning": "<2-3 sentences, specific and evidence-based>",
  "recommendation": "<strong_fit|moderate_fit|weak_fit>"}}"""


def map_prompt(rubric, chunk_text: str, section: str) -> str:
    return f"""RUBRIC CATEGORIES: {", ".join(n for n, _, _, _ in rubric._prepared)}

RESUME EXCERPT (section: {section}):
\"\"\"{chunk_text}\"\"\"

Extract ONLY what is explicitly present in this excerpt.
OUTPUT a single JSON object:
{{"evidence": ["<max 5 short quoted phrases showing relevant skill/experience>"],
  "skills": ["<max 10 concrete skills/tools named in the excerpt>"],
  "years_signal": <number or 0>}}"""


def reduce_prompt(jd_text: str, rubric, evidence: list[str], skills: list[str],
                  matched: list[str], missing: list[str]) -> str:
    ev = "\n".join(f"- {e}" for e in evidence[:40]) or "none"
    sk = ", ".join(sorted(set(skills))[:60]) or "none"
    return f"""JOB DESCRIPTION:
\"\"\"{jd_text[:2500]}\"\"\"

RUBRIC WEIGHTS:
{_weights_block(rubric)}

EVIDENCE EXTRACTED FROM THE CANDIDATE'S RESUME (multi-page, chunked):
{ev}

SKILLS NAMED: {sk}
Keyword pre-scan found: {", ".join(matched[:30]) or "none"}
Keyword pre-scan did NOT find: {", ".join(missing[:20]) or "none"}

TASK: Score this candidate 0-100 for the role using only the evidence above.
OUTPUT a single JSON object:
{{"score": <integer 0-100>,
  "confidence": <float 0.0-1.0>,
  "years_relevant": <number>,
  "seniority": "<intern|junior|mid|senior|lead>",
  "strengths": ["<max 3 items, grounded in the evidence>"],
  "gaps": ["<max 3 items>"],
  "reasoning": "<2-3 sentences, specific and evidence-based>",
  "recommendation": "<strong_fit|moderate_fit|weak_fit>"}}"""
