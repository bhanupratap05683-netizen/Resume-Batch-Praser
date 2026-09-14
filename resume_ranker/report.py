"""Output writers: ranked CSV (+ top-N CSV), JSON, and an HTML dashboard."""
from __future__ import annotations

import csv
import html
from datetime import datetime
from pathlib import Path

from .pipeline import Candidate

COLUMNS: list[tuple[str, str]] = [
    ("rank", "Rank"),
    ("file", "File"),
    ("name", "Candidate"),
    ("email", "Email"),
    ("phone", "Phone"),
    ("linkedin", "LinkedIn"),
    ("github", "GitHub"),
    ("location", "Location"),
    ("final_score", "Final Score"),
    ("det_score", "Keyword Score"),
    ("llm_score", "LLM Score"),
    ("recommendation", "Recommendation"),
    ("years_experience", "Years Exp"),
    ("education_level", "Education"),
    ("education_fields", "Fields"),
    ("seniority", "Seniority"),
    ("must_have_met", "Must-Have Met"),
    ("must_have_missing", "Must-Have Missing"),
    ("top_skills", "Top Matched Skills"),
    ("missing_skills", "Missing Skills"),
    ("strengths", "LLM Strengths"),
    ("gaps", "LLM Gaps"),
    ("reasoning", "Reasoning"),
    ("evidence", "Keyword Evidence"),
    ("ocr_used", "OCR Used"),
    ("parse_status", "Parse Status"),
    ("warnings", "Warnings"),
]


def _rows(cands: list[Candidate]) -> list[dict]:
    rows = []
    for c in cands:
        rows.append({
            "rank": c.rank,
            "file": c.file,
            "name": c.name,
            "email": c.email,
            "phone": c.phone,
            "linkedin": c.linkedin,
            "github": c.github,
            "location": c.location,
            "final_score": f"{c.final_score:.2f}",
            "det_score": f"{c.det_calibrated:.2f}",
            "llm_score": "" if c.llm_score is None else f"{c.llm_score:.2f}",
            "recommendation": c.recommendation,
            "years_experience": c.years_experience,
            "education_level": c.education_level,
            "education_fields": "; ".join(c.education_fields),
            "seniority": c.seniority,
            "must_have_met": "; ".join(c.must_have_met),
            "must_have_missing": "; ".join(c.must_have_missing),
            "top_skills": "; ".join(c.matched_skills[:18]),
            "missing_skills": "; ".join(c.missing_skills[:12]),
            "strengths": " | ".join(c.strengths),
            "gaps": " | ".join(c.gaps),
            "reasoning": c.reasoning,
            "evidence": " | ".join(c.evidence[:6]),
            "ocr_used": "yes" if c.ocr_used else "no",
            "parse_status": c.parse_status,
            "warnings": "; ".join(c.warnings),
        })
    return rows


def write_csv(cands: list[Candidate], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        # Friendly column titles for humans, stable internal keys for code.
        w.writerow([label for _, label in COLUMNS])
        for row in _rows(cands):
            w.writerow([row.get(key, "") for key, _ in COLUMNS])
    return path


def write_top_csv(cands: list[Candidate], path: Path, top_n: int) -> Path:
    return write_csv([c for c in cands if c.rank <= top_n], path)


def write_json(cands: list[Candidate], path: Path) -> Path:
    import json
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(c) for c in cands], indent=2, default=str), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------
# HTML dashboard (self-contained: inline CSS, no CDN -- works offline)
# --------------------------------------------------------------------------
def _bar(score: float) -> str:
    pct = max(0.0, min(100.0, score))
    color = "#16a34a" if pct >= 75 else ("#d97706" if pct >= 58 else "#dc2626")
    return (f'<div class="bar"><div class="fill" style="width:{pct:.0f}%;'
            f'background:{color}"></div><span>{score:.1f}</span></div>')


def write_html(cands: list[Candidate], path: Path, top_n: int, meta: dict) -> Path:
    esc = html.escape
    rows = []
    for c in cands:
        if c.rank > top_n:
            continue
        reason = esc(c.llm_reasoning or c.reasoning or "")
        rows.append(f"""<tr>
<td class="rank">{c.rank}</td>
<td><div class="nm">{esc(c.name or '(unknown)')}</div>
    <div class="mut">{esc(c.email or 'no email')} · {esc(c.phone or 'no phone')}</div>
    <div class="mut">{esc(c.file)} · {esc(c.location)}</div></td>
<td>{_bar(c.final_score)}<div class="mut">kw {c.det_calibrated:.0f} · llm {'—' if c.llm_score is None else f'{c.llm_score:.0f}'} · {esc(c.recommendation)}</div></td>
<td>{esc(str(c.years_experience or 0))}y<div class="mut">{esc(c.education_level or "")}</div></td>
<td><div class="tags">{''.join(f'<span class="t ok">{esc(s)}</span>' for s in c.matched_skills[:10])}</div></td>
<td><div class="tags">{''.join(f'<span class="t no">{esc(s)}</span>' for s in c.missing_skills[:6])}</div></td>
<td class="reason">{reason}</td>
</tr>""")

    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Resume Ranking — {esc(meta.get('jd_title',''))}</title>
<style>
body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#111}}
.wrap{{max-width:1500px;margin:0 auto;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#666;font-size:13px;margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
th{{background:#1f2937;color:#fff;text-align:left;padding:10px 12px;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
td{{padding:10px 12px;border-top:1px solid #eef0f2;vertical-align:top}}
.rank{{font-size:18px;font-weight:700;color:#374151;width:44px}}
.nm{{font-weight:600}} .mut{{color:#6b7280;font-size:12px}}
.bar{{position:relative;background:#e5e7eb;border-radius:6px;height:20px;min-width:120px}}
.bar .fill{{height:100%;border-radius:6px}}
.bar span{{position:absolute;right:8px;top:1px;font-size:12px;font-weight:700;color:#111}}
.tags{{display:flex;flex-wrap:wrap;gap:4px;max-width:290px}}
.t{{font-size:11px;padding:2px 7px;border-radius:99px}}
.t.ok{{background:#dcfce7;color:#166534}} .t.no{{background:#fee2e2;color:#991b1b}}
.reason{{max-width:520px;font-size:12.5px;color:#374151}}
.meta{{margin-top:14px;color:#6b7280;font-size:12px}}
</style></head><body><div class="wrap">
<h1>Top {min(top_n,len(cands))} — {esc(meta.get('jd_title','Ranked Candidates'))}</h1>
<div class="sub">Scored {meta.get('total',0)} resumes · generated {gen} · model {esc(meta.get('model','n/a'))}</div>
<table><thead><tr><th>#</th><th>Candidate</th><th>Score</th><th>Exp</th><th>Matched</th><th>Gaps</th><th>Reasoning</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<div class="meta">Final = {meta.get('w_det')}×keyword + {meta.get('w_llm')}×LLM judge · keyword score is deterministic rubric coverage; LLM score is qwen2.5-coder's evidence-based judgement.</div>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
