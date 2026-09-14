#!/usr/bin/env python3
"""Resume batch parser + JD scorer + ranker, powered by local Ollama.

Usage
-----
    python main.py audit  --pdf-dir uploads
    python main.py check  --model qwen2.5-coder:7b
    python main.py run    --pdf-dir uploads --jd jd.txt --top 10
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resume_ranker import report as report_mod  # noqa: E402
from resume_ranker.config import Config  # noqa: E402
from resume_ranker.extract import ExtractConfig, _sha1_of  # noqa: E402
from resume_ranker.pipeline import run, parse_deterministic  # noqa: E402
from resume_ranker.rubric import load_rubric  # noqa: E402

log = logging.getLogger("resume_ranker")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("rapidocr_onnxruntime").setLevel(logging.ERROR)


# --------------------------------------------------------------------------
def cmd_audit(args: argparse.Namespace) -> int:
    """Report whether each PDF has a real text layer or needs OCR."""
    import pymupdf

    pdfs = sorted(Path(args.pdf_dir).glob("**/*.pdf"),
                  key=lambda p: (len(p.stem), p.stem))
    if not pdfs:
        print(f"No PDFs found in {args.pdf_dir}")
        return 1
    print(f"{'file':<26}{'pgs':>4}{'chars':>8}{'words':>7}{'imgs':>6}  {'text layer':<14} note")
    print("-" * 78)
    stats = {"text": 0, "scan": 0, "empty": 0}
    for p in pdfs:
        try:
            doc = pymupdf.open(p)
            txt, imgs = "", 0
            for pg in doc:
                txt += pg.get_text()
                imgs += len(pg.get_images(full=True))
            n = doc.page_count
            doc.close()
            alpha = sum(c.isalpha() for c in txt)
            if alpha >= 200 * max(1, n):
                kind, note, stats_key = "YES", "", "text"
            elif alpha > 0:
                kind, note, stats_key = "PARTIAL", "will OCR", "scan"
            else:
                kind, note, stats_key = "NO (scanned)", "will OCR", "scan"
            if not txt.strip() and not imgs:
                note, stats_key = "no content!", "empty"
            stats[stats_key] += 1
            print(f"{p.name:<26}{n:>4}{len(txt):>8}{len(txt.split()):>7}{imgs:>6}  {kind:<14} {note}")
        except Exception as exc:
            print(f"{p.name:<26}{'':>4}{'':>8}{'':>7}{'':>6}  ERROR           {exc}")
    print("-" * 78)
    print(f"{len(pdfs)} PDF(s): {stats['text']} with text layer, "
          f"{stats['scan']} need OCR, {stats['empty']} broken")
    if stats["scan"]:
        print("\nOCR will be used. For speed install:  pip install rapidocr-onnxruntime")
    return 0


# --------------------------------------------------------------------------
def cmd_check(args: argparse.Namespace) -> int:
    """Verify Ollama is reachable and the model is pulled."""
    import httpx

    host = args.host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as exc:
        print(f"✗ Cannot reach Ollama at {host}: {exc}")
        print("  Start it with:  ollama serve")
        return 1
    models = [m.get("name", "") for m in r.json().get("models", [])]
    print(f"✓ Ollama reachable at {host}")
    print(f"  Pulled models: {', '.join(models) or '(none)'}")
    ok = any(m.split(':')[0] == args.model.split(':')[0] for m in models)
    print(f"  {'✓' if ok else '✗'} model '{args.model}'")
    if not ok:
        print(f"\n  Run:  ollama pull {args.model}")
        return 1

    # Live speed probe.
    print("\nRunning a tiny generation to measure throughput...")
    import time
    t0 = time.perf_counter()
    rr = httpx.post(f"{host.rstrip('/')}/api/chat", json={
        "model": args.model, "stream": False, "keep_alive": "10m",
        "options": {"num_ctx": 2048, "num_predict": 120, "temperature": 0},
        "messages": [{"role": "user", "content": "Write a 3 sentence summary of what a Sharpe ratio is."}],
    }, timeout=180)
    el = time.perf_counter() - t0
    data = rr.json()
    tok = data.get("eval_count", 0)
    tps = tok / (data.get("eval_duration", 1) / 1e9) if data.get("eval_duration") else 0
    print(f"  {tok} tokens in {el:.1f}s  ({tps:.1f} tok/s)")
    print(f"  → roughly {60 / max(0.4, el):.0f} resumes/min at this speed (1 call each)")
    print("\nTip: export OLLAMA_NUM_PARALLEL=4 before `ollama serve` to allow concurrency.")
    return 0


# --------------------------------------------------------------------------
def _preflight_ram_check(pdfs: list[Path], cfg: Config) -> None:
    """Warn before a batch that could exhaust RAM.

    OCR (~1.2 GB/worker at 1700px) and Ollama (~5 GB for a 7B Q4_K_M) do NOT
    overlap on a cold start, because extraction completes before the model is
    first contacted. They DO overlap if a previous run left the model resident
    (keep_alive=-1) or if you pre-warmed it -- so check free memory.
    """
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total_gb, avail_gb = vm.total / (1024**3), vm.available / (1024**3)
    except Exception:
        return

    try:
        from .extract import _auto_workers

        workers = _auto_workers(cfg.extract)
    except Exception:
        workers = cfg.extract.ocr_workers or max(1, min(os.cpu_count() or 2, 4))
    ocr_gb = workers * 1.2
    model_gb = 5.0 if cfg.use_llm else 0.0
    worst = ocr_gb + model_gb  # warm-start case

    print(f"RAM: {avail_gb:.1f} GB free of {total_gb:.1f} GB | "
          f"OCR workers={workers} (~{ocr_gb:.1f} GB)"
          + (f" | model ~{model_gb:.0f} GB" if model_gb else ""))
    if worst > avail_gb - 2.0:
        print(f"  !! Tight: worst case ~{worst:.1f} GB vs {avail_gb:.1f} GB free.")
        print("     Reduce load:  --ocr-workers 2  --concurrency 2")
    elif model_gb and avail_gb < 8.0:
        print("  ! Under 8 GB free: consider --ocr-workers 2 --concurrency 2")
    if cfg.use_llm and cfg.ollama.concurrency > 2:
        print("  tip: set OLLAMA_NUM_PARALLEL to match --concurrency "
              f"({cfg.ollama.concurrency}) before `ollama serve`.")
    print()


# --------------------------------------------------------------------------
async def _run_async(args: argparse.Namespace, cfg: Config) -> int:
    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("**/*.pdf"), key=lambda p: (len(p.stem), p.stem))
    if not pdfs:
        print(f"No PDFs found under {pdf_dir}")
        return 1
    if args.limit:
        pdfs = pdfs[: args.limit]

    _preflight_ram_check(pdfs, cfg)

    jd_text = Path(args.jd).read_text(encoding="utf-8") if args.jd else ""
    rubric = load_rubric(args.rubric)

    print(f"\n{'=' * 70}\n  {rubric.title}\n  {len(pdfs)} PDFs · model={cfg.ollama.model} "
          f"· llm={'on' if cfg.use_llm else 'OFF (keyword only)'}\n{'=' * 70}\n")

    done = {"n": 0}

    def progress(msg: str) -> None:
        done["n"] += 1
        if done["n"] % 5 == 0 or done["n"] == len(pdfs):
            print(f"  ...{done['n']}/{len(pdfs) * (2 if cfg.use_llm else 1)} {msg}")

    cands = await run(pdfs, rubric, jd_text, cfg, progress=progress if not args.quiet else None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "all CSV": report_mod.write_csv(cands, out / "ranked_all.csv"),
        f"top {args.top} CSV": report_mod.write_top_csv(cands, out / f"top_{args.top}.csv", args.top),
        "full JSON": report_mod.write_json(cands, out / "parsed_candidates.json"),
        "dashboard": report_mod.write_html(cands, out / "report.html", args.top, {
            "jd_title": rubric.title, "total": len(cands), "model": cfg.ollama.model,
            "w_det": cfg.score.weight_deterministic, "w_llm": cfg.score.weight_llm,
        }),
    }

    print(f"\n{'=' * 70}\nRANKED RESULTS (top {min(args.top, len(cands))})\n{'=' * 70}")
    hdr = f"{'#':>3}  {'score':>6} {'kw':>5} {'llm':>5}  {'candidate':<24} {'email':<28} rec"
    print(hdr)
    print("-" * 100)
    for c in cands[: args.top]:
        llm_s = "  -  " if c.llm_score is None else f"{c.llm_score:5.1f}"
        print(f"{c.rank:>3}  {c.final_score:6.2f} {c.det_calibrated:5.1f} {llm_s}  "
              f"{(c.name or '(unknown)')[:24]:<24} {(c.email or '-')[:28]:<28} {c.recommendation}")

    print(f"\nOutputs written to {out.resolve()}:")
    for label, p in files.items():
        print(f"  {label:<12} {p}")
    n_ocr = sum(1 for c in cands if c.ocr_used)
    n_err = sum(1 for c in cands if c.parse_status != "ok")
    print(f"\n{n_ocr}/{len(cands)} required OCR · {n_err} had parse problems")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config()
    cfg.top_n = args.top
    cfg.use_llm = not args.no_llm
    cfg.ollama.host = args.host or os.environ.get("OLLAMA_HOST", cfg.ollama.host)
    cfg.ollama.model = args.model
    cfg.ollama.concurrency = args.concurrency
    if args.num_ctx:
        cfg.ollama.chunk_ctx = cfg.ollama.reduce_ctx = args.num_ctx
    cfg.extract.ocr_enabled = not args.no_ocr
    cfg.extract.ocr_force = args.force_ocr
    cfg.extract.cache_dir = args.cache_dir
    cfg.extract.use_cache = not args.no_cache
    if args.ocr_dpi:
        cfg.extract.ocr_dpi = args.ocr_dpi
    if args.ocr_target:
        cfg.extract.ocr_target_long_side = args.ocr_target
    if args.ocr_workers:
        cfg.extract.ocr_workers = args.ocr_workers
    cfg.chunk.max_chunks_to_llm = args.max_chunks
    cfg.score.weight_deterministic = args.w_det
    cfg.score.weight_llm = args.w_llm

    if args.no_cache:
        print("(cache disabled)")
    return asyncio.run(_run_async(args, cfg))


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch resume parser + JD scorer (local Ollama)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="check whether PDFs have a text layer or need OCR")
    a.add_argument("--pdf-dir", default="uploads")
    a.set_defaults(func=cmd_audit)

    c = sub.add_parser("check", help="verify Ollama + model and measure speed")
    c.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b"))
    c.add_argument("--host", default=None)
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="parse, score and rank resumes")
    r.add_argument("--pdf-dir", default="uploads")
    r.add_argument("--jd", default="jd.txt")
    r.add_argument("--rubric", default="rubrics/financial_data_analyst.json")
    r.add_argument("--out", default="out")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--limit", type=int, default=0, help="only process first N PDFs")
    r.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b"))
    r.add_argument("--host", default=None)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--num-ctx", type=int, default=0)
    r.add_argument("--max-chunks", type=int, default=6)
    r.add_argument("--w-det", type=float, default=0.40)
    r.add_argument("--w-llm", type=float, default=0.60)
    r.add_argument("--no-llm", action="store_true", help="keyword scoring only (no Ollama)")
    r.add_argument("--no-ocr", action="store_true")
    r.add_argument("--force-ocr", action="store_true", help="OCR even if a text layer exists")
    r.add_argument("--ocr-dpi", type=int, default=0)
    r.add_argument("--ocr-target", type=int, default=0, help="long-side pixels for OCR render")
    r.add_argument("--ocr-workers", type=int, default=0)
    r.add_argument("--cache-dir", default="cache")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)
    return p


def main() -> int:
    args = build_parser().parse_args()
    _setup_logging(getattr(args, "verbose", False))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
