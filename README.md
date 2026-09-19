# Resume Batch Parser → JD Scorer → Top-10 Ranker (local Ollama)

Parses a folder of PDF resumes, scores each against a job description, and
writes a ranked CSV with **scores and reasoning**. Runs fully offline on your
machine with **Ollama + qwen2.5-coder:7b**.

```
folder of PDFs ──▶ extract ──▶ parse ──▶ chunk ──▶ score ──▶ rank ──▶ CSV / HTML
  (any count)      text/OCR   contacts   sections   keyword    blend    + JSON
                                                    + LLM
```

Design goals: **correct on scanned PDFs**, **fast on 50+ files**, and
**explainable** — every score carries the evidence that produced it.

---

## TL;DR — the four commands

```bash
# 0. once: install deps and pull the model
pip install -r requirements.txt
ollama pull qwen2.5-coder:7b

# 1. are the PDFs text-based or scanned?
python main.py audit --pdf-dir uploads

# 2. is Ollama alive and how fast is it?
python main.py check --model qwen2.5-coder:7b

# 3. parse → score → rank
python main.py run --pdf-dir uploads --jd jd.txt --top 10
```

Outputs land in `out/` (or `--out DIR`): `ranked_all.csv`, `top_10.csv`,
`parsed_candidates.json`, `report.html`.

---

## Step-by-step guide

### Step 0 — Start Ollama the *fast* way

The single biggest speed factor is Ollama's own concurrency setting. Start the
server with these environment variables:

```bash
# Linux / macOS
export OLLAMA_NUM_PARALLEL=4        # concurrent requests (match --concurrency)
export OLLAMA_MAX_LOADED_MODELS=1   # keep just this model resident
export OLLAMA_FLASH_ATTENTION=1     # lower VRAM, faster prefill (GPU builds)
export OLLAMA_KEEP_ALIVE=-1         # never unload between runs
ollama serve

# Windows PowerShell
setx OLLAMA_NUM_PARALLEL 4
setx OLLAMA_KEEP_ALIVE -1
ollama serve
```

Then pull the model once:

```bash
ollama pull qwen2.5-coder:7b
ollama list          # confirm it is there
```

> **Which tag?** Use a quantized instruct build for speed. Rough guide on a
> 7B: `q4_K_M` ≈ fastest/lowest RAM, `q5_K_M` ≈ best quality/speed balance,
> `q8_0` ≈ near-fp16 quality, ~2× slower. If you pulled a different tag, pass
> it with `--model qwen2.5-coder:7b-instruct-q5_K_M`.

### Step 1 — Install Python dependencies

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`pymupdf` reads PDFs; `rapidocr-onnxruntime` is the OCR engine (pure pip, no
system packages — this is why it is the default). If you prefer Tesseract:
`sudo apt install tesseract-ocr && pip install pytesseract` — it is auto-detected.

### Step 2 — Check the PDFs *first* (do not skip this)

```bash
python main.py audit --pdf-dir uploads
```

```
file        pgs  chars words imgs  text layer     note
1.pdf         1     0     0    1  NO (scanned)   will OCR
2.pdf         1     0     0    1  NO (scanned)   will OCR
...
20 PDF(s): 0 with text layer, 20 need OCR, 0 broken
```

This tells you whether OCR will be needed and how long the batch will take.
**Your 20 files are all single-page scans with no text layer**, so every one
goes through OCR (~4–10 s each on CPU, then cached forever).

### Step 3 — Verify Ollama and measure throughput

```bash
python main.py check --model qwen2.5-coder:7b
```

It confirms the server is reachable, the model is pulled, and prints a live
tokens/second figure so you can predict batch time.

### Step 4 — Run the pipeline

```bash
python main.py run --pdf-dir uploads --jd jd.txt --top 10
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--out DIR` | output directory (default `out`) |
| `--top N` | also write `top_N.csv` and limit the HTML table |
| `--no-llm` | keyword scoring only, no Ollama (great for a first pass) |
| `--concurrency N` | parallel LLM requests — **keep ≤ `OLLAMA_NUM_PARALLEL`** |
| `--model TAG` | Ollama model tag |
| `--w-det 0.4 --w-llm 0.6` | blend weights between keyword and LLM score |
| `--limit N` | process only the first N PDFs (smoke test) |
| `--force-ocr` | ignore the text layer and OCR anyway |
| `--ocr-target 2200` | OCR render size in px (bigger = better on dense pages) |
| `--ocr-workers N` | parallel OCR processes |
| `--no-cache` | re-extract everything, ignoring the cache |
| `--max-chunks N` | max chunks of a long CV sent to the LLM |

**Recommended first run** (cheap, catches data problems before you spend GPU time):

```bash
python main.py run --pdf-dir uploads --limit 3 --no-llm --out /tmp/smoke
```

### Step 5 — Read the results

`ranked_all.csv` is the deliverable — one row per candidate, 27 columns:

| Column | What it tells you |
|---|---|
| `Final Score` | blended 0–100 (default `0.4 × keyword + 0.6 × LLM`) |
| `Keyword Score` | deterministic rubric coverage (reproducible, no model) |
| `LLM Score` | qwen2.5-coder's judgement |
| `Reasoning` | why: must-haves met/missing, experience bonus, model's reasoning |
| `LLM Strengths` / `LLM Gaps` | 3 bullets each, grounded in resume text |
| `Keyword Evidence` | the exact resume snippets that matched, per rubric category |
| `Must-Have Met` / `Must-Have Missing` | hard requirements from the rubric |
| `Email` / `Phone` / `LinkedIn` | regex-extracted (never from the LLM) |
| `Warnings` | `placeholder-email`, `duplicate-of:15.pdf`, `scorer-disagreement`, … |

Also written: `top_10.csv`, `parsed_candidates.json` (everything, machine-readable)
and `report.html` (self-contained dashboard — open it directly in a browser).

---

## How the scoring works

**Two independent scorers are blended**, because each covers the other's blind spot.

**1. Deterministic rubric (`rubrics/financial_data_analyst.json`)** — 10 weighted
categories summing to 100:

| Category | Weight | Must-have |
|---|---|---|
| Statistical & Quantitative Methods | 20 | ✅ |
| Core Programming (Python/R) | 18 | ✅ |
| Financial Domain Knowledge | 18 | ✅ |
| SQL & Data Engineering | 14 | ✅ |
| Advanced Excel & Financial Modeling | 8 | |
| BI & Dashboards | 6 | |
| Machine Learning & Factor Modeling | 5 | |
| Cloud & Engineering Practice | 4 | |
| Professional Credentials | 4 | |
| Market Data & Terminals | 3 | |

Matching uses **squashed text** (letters+digits only, lowercased), which makes
it immune to OCR damage: `Power BI` ≡ `PowerBI` ≡ `Power-BI` ≡ `Power Bl`.

**Precision rule (important).** Aliases whose squashed form is **shorter than 5
characters are matched on word boundaries in the raw text**, not as substrings.
Short substrings are a false-positive factory — verified on real OCR output:

| alias | fired inside | verdict |
|---|---|---|
| `r` | every English sentence | fabricated R language |
| `ols` | "t**ols**" | fabricated regression |
| `arch` | "rese**arch**" | fabricated ARCH/GARCH |
| `cte` | "imp**acte**d" | fabricated advanced SQL |
| `roc` | "p**roc**ess" | fabricated ML practice |
| `api` | "r**api**dly" | fabricated API skill |
| `dash` | "**dash**boards" | fabricated Dash/Streamlit |

Raise/lower the threshold with `Rubric.min_substring_len` (default `5`).
Long strings (≥5) essentially never occur inside an unrelated word, so they
keep the OCR-robust substring behaviour.
Each category gets diminishing returns (`1 − e^(−3·coverage)`), then we add an
experience bonus and subtract penalties for missing must-haves.

**2. LLM judge (qwen2.5-coder:7b)** — sees the JD, the rubric weights, the
most relevant chunks, *and* the keyword pre-scan, then returns JSON with a
score, 3 strengths, 3 gaps and a 2–3 sentence reasoning.

Final = `0.4 × keyword + 0.6 × LLM`. When the two disagree by >25 points the
row is flagged `scorer-disagreement` — a useful "human should look here" signal.

**Contacts are never LLM-extracted.** Emails, phones, LinkedIn/GitHub and
locations come from OCR-hardened regexes (`resume_ranker/contacts.py`): a
7B model hallucinates digits in phone numbers and silently "cleans" emails.
The regexes repair the specific damage scanners cause —
`john.doe @ gmail .com` → `john.doe@gmail.com`,
`555-555-5555-example@example.com` → `example@example.com`,
`ROBERTSMITH` → `Robert Smith`, `linkedln.com` → `linkedin.com`.

---

## Why it is fast

| Technique | Where | Effect |
|---|---|---|
| **OCR cache** keyed by file hash | `cache/` | re-runs cost 0 s of OCR |
| **Cheapest-first extraction** | `extract.py` | text layer used when present; OCR only where needed |
| **Process pool** for OCR | `extract.py` | CPU-bound work escapes the GIL |
| **Memory-adaptive render** | `extract.py` | auto-shrinks scan resolution on small machines instead of OOM-ing |
| **Async + semaphore** | `llm.py` | N resumes in flight; never exceeds `OLLAMA_NUM_PARALLEL` |
| **Chunking + relevance ranking** | `chunk.py` | only the top-6 JD-relevant chunks reach the model |
| **Small context** (`num_ctx 4096`) | `llm.py` | far faster prefill than Ollama's 32k default |
| **`num_predict` caps** | `llm.py` | one runaway answer can't stall the batch |
| **`keep_alive=-1` + warmup** | `llm.py` | model stays resident; no per-file reload |
| **JSON mode + repair + retry** | `llm.py` | constrained decoding; bad output retried, never fatal |

Measured on this batch in a 2 GB sandbox: 20 scanned PDFs OCR'd once (~9 s
each), then 20 parallel LLM judgements in ~4 s of wall time.

---

## Adapting it to another job description

1. Put the new JD in `jd.txt` (or `--jd other.txt`).
2. Edit `rubrics/financial_data_analyst.json`:
   - rename `title`
   - adjust category `weight`s so they sum to 100
   - replace the `skills` lists — each entry is a *canonical name* plus the
     aliases that should count as a match:
     ```json
     {"name": "Power BI", "aliases": ["Power BI", "PowerBI", "Power-BI", "DAX"]}
     ```
   - set `must_have: true` on categories that are genuinely disqualifying
   - update `education_relevant_fields` and `experience_target_years`
3. Re-run. No code changes needed.

---

## Troubleshooting

### Memory on a 16 GB machine

The two heavy stages are **sequential, not concurrent**: extraction finishes
and the OCR process pool is torn down *before* Ollama is first contacted. So a
cold run peaks at roughly `max(OCR, Ollama)`, not their sum.

Measured ONNX peak per OCR worker: **0.67 GB @1100px, 0.85 GB @1300px,
1.19 GB @1600px**. Worker count is now budgeted against *free* RAM
(`_auto_workers`), reserving ~2 GB headroom.

| Scenario | Peak | Verdict on 16 GB |
|---|---|---|
| Cold start (no model resident) | ~4.6 GB OCR → then ~6 GB model | comfortable |
| Warm start (model left resident by `keep_alive=-1`) | ~4.6 + ~5 + ~4 GB OS ≈ 13.6 GB | tight — swap risk if a browser is open |

For the warm case (the common one — you *want* `keep_alive=-1`), use:

```bash
python main.py run --pdf-dir uploads --jd jd.txt --top 10 --concurrency 1 --ocr-workers 2
```

`main.py` prints a preflight RAM report and warns you before starting a batch
that would not fit. The first run pays for OCR; every later run loads text from
`cache/` and leaves the whole machine to Ollama.

| Symptom | Fix |
|---|---|
| `Cannot reach Ollama` | start it: `ollama serve`; check `--host` / `OLLAMA_HOST` |
| `Model not pulled` | `ollama pull qwen2.5-coder:7b` |
| Everything scores 0 | run `python main.py audit` — if `text layer = NO`, install an OCR backend |
| `No OCR backend available` | `pip install rapidocr-onnxruntime` |
| OCR output is garbage | raise resolution: `--ocr-target 2200` (or `--ocr-dpi 300`) |
| OCR process gets killed | lower resolution: `--ocr-target 1200 --ocr-workers 1` |
| LLM calls are slow | raise `OLLAMA_NUM_PARALLEL`, then `--concurrency` to match; use a `q4_K_M` tag |
| Scores look random | check `Warnings` for `scorer-disagreement`; try `--w-det 0.6 --w-llm 0.4` to trust keywords more |
| `placeholder-email` | template sample address (qwikresume/enhancv/example.com) — real candidate contact is missing |
| `duplicate-of:15.pdf` | byte-identical PDF, scored twice on purpose so you can dedupe |
| Re-run after a crash | just re-run; the cache skips already-extracted files |

---

## Layout

```
main.py                     CLI: audit | check | run
jd.txt                      the job description
rubrics/
  financial_data_analyst.json   weighted skill rubric (edit this for a new JD)
resume_ranker/
  config.py                 tunables (Ollama, OCR, chunking, scoring)
  extract.py                PDF → text (text layer, then OCR, then cache)
  contacts.py               OCR-hardened email/phone/name/location regexes
  textutils.py              squashing, section detection, snippets
  chunk.py                  section-aware chunking + relevance ranking
  rubric.py                 deterministic weighted scoring + education/exp
  prompts.py                prompt templates tuned for 7B models
  llm.py                    async Ollama client (semaphore, retry, JSON repair)
  score.py                  calibration and keyword/LLM blending
  pipeline.py               orchestration (extract → parse → judge → rank)
  report.py                 CSV / JSON / HTML writers
  ocr_worker.py             one-shot OCR subprocess (low-RAM machines)
tools/
  mock_ollama.py            fake Ollama server for testing without a model
cache/                      extracted text, keyed by SHA-1 (safe to delete)
out/                        results
```

## Testing without a GPU or a model

```bash
python tools/mock_ollama.py --port 11555
python main.py run --pdf-dir uploads --host http://127.0.0.1:11555 --concurrency 4 --out /tmp/mock
```

This exercises the whole async path (batching, JSON parsing, retries,
blending) with a stub that invents plausible scores.
