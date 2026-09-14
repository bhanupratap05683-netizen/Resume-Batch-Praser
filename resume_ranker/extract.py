"""PDF -> text.

Three-stage strategy, cheapest first:

1. **Native text layer** (PyMuPDF). ~1 ms/page, perfect fidelity. Most
   digitally-generated resumes hit this path.
2. **OCR** of a re-rendered page. Used when the text layer is empty or
   sparse (scanned / image-only PDFs) -- which is the case for every file
   in this batch.
3. **Cache**: results are keyed by file content hash, so re-running the
   pipeline (e.g. after tuning the rubric) never pays for OCR twice.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .config import ExtractConfig

log = logging.getLogger(__name__)

_WS = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL = re.compile(r"\n{3,}")

# --------------------------------------------------------------------------
# OCR backends. Instantiated lazily and cached *per process* (important: the
# OCR engine is expensive to build and is NOT fork-safe to share).
# --------------------------------------------------------------------------
_ENGINE: dict[str, Any] = {"name": None, "obj": None}
_ENGINE_LOCK = threading.Lock()


def _build_engine(backends: tuple[str, ...]) -> tuple[str | None, Any]:
    for name in backends:
        try:
            if name == "rapidocr":
                from rapidocr_onnxruntime import RapidOCR  # type: ignore

                # NB: do NOT pass det_limit_side_len / det_limit_type here --
                # several rapidocr-onnxruntime versions then fail to resolve
                # their bundled model_path (KeyError: 'model_path'). Control
                # the input size by rendering the page at the DPI we want
                # instead (see _render_dpi).
                try:
                    eng = RapidOCR(intra_op_num_threads=1)
                except TypeError:
                    eng = RapidOCR()
                return "rapidocr", eng
            if name == "tesseract":
                import pytesseract  # type: ignore
                from PIL import Image  # type: ignore

                pytesseract.get_tesseract_version()
                return "tesseract", (pytesseract, Image)
            if name == "paddleocr":
                from paddleocr import PaddleOCR  # type: ignore

                return "paddleocr", PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception as exc:  # pragma: no cover - depends on environment
            log.debug("OCR backend %s unavailable: %s", name, exc)
    return None, None


def get_engine(backends: tuple[str, ...] = ("rapidocr", "tesseract", "paddleocr")):
    with _ENGINE_LOCK:
        if _ENGINE["obj"] is None:
            name, obj = _build_engine(backends)
            _ENGINE["name"], _ENGINE["obj"] = name, obj
            if name is None:
                log.warning(
                    "No OCR backend available. Install one of: "
                    "pip install rapidocr-onnxruntime | apt install tesseract-ocr && pip install pytesseract"
                )
        return _ENGINE["name"], _ENGINE["obj"]


def _ocr_image(obj: Any, backend: str, img_bytes: bytes) -> str:
    if backend == "rapidocr":
        result, _ = obj(img_bytes)
        if not result:
            return ""
        return "\n".join(line[1] for line in result)
    if backend == "tesseract":
        import io

        pytesseract, Image = obj
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img) or ""
    if backend == "paddleocr":
        import io
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        res = obj.ocr(np.array(img), cls=True)
        out: list[str] = []
        for block in res or []:
            for line in block or []:
                out.append(line[1][0])
        return "\n".join(out)
    return ""


_RAM_TARGET_CACHE: dict[int, int] = {}


def auto_target_long_side(requested: int) -> int:
    """Shrink the render size on small machines instead of dying with an OOM.

    OCR peak memory scales with image area. A 1700px-long-side page is
    comfortable on a 16 GB laptop but can exceed a 2 GB container, so we
    scale the target down rather than letting the OS kill the process.
    """
    try:
        import psutil  # type: ignore

        gb = int(psutil.virtual_memory().total / (1024**3))
    except Exception:
        gb = 8
    if gb in _RAM_TARGET_CACHE:
        base = _RAM_TARGET_CACHE[gb]
    else:
        if gb <= 2:
            base = 1100
        elif gb <= 4:
            base = 1400
        elif gb <= 8:
            base = 1700
        else:
            base = 2000
        _RAM_TARGET_CACHE[gb] = base
    return min(requested, base) if requested else base


def _render_dpi(page_rect, cfg: ExtractConfig, native_px: int | None) -> int:
    """DPI that lands the page's long side near `ocr_target_long_side`.

    Low-res scans get upscaled (OCR needs ~20px of height per text line);
    300-DPI A4 scans get downscaled so we don't burn RAM for nothing.
    """
    if cfg.ocr_dpi:
        return int(cfg.ocr_dpi)
    long_side_pt = max(page_rect.width, page_rect.height) or 792.0
    target = auto_target_long_side(cfg.ocr_target_long_side)
    dpi_needed = 72.0 * target / long_side_pt

    # Upscaling tiny scans helps OCR a lot (it needs ~20 px of text height),
    # but never past the pixel budget set by `target` -- on a small machine
    # an unconditional 200-DPI floor is what actually causes the OOM.
    dpi = max(dpi_needed, float(cfg.ocr_min_dpi))
    if long_side_pt * dpi / 72.0 > target * 1.25:
        dpi = dpi_needed
    return int(min(float(cfg.ocr_max_dpi), dpi))


def _page_text(doc, page_index: int, cfg: ExtractConfig, force_ocr: bool) -> tuple[str, bool]:
    """Return (text, ocr_used) for a single page."""
    import pymupdf# imported lazily so workers don't all pay import cost

    page = doc[page_index]
    if not force_ocr:
        native = page.get_text("text") or ""
        if sum(c.isalpha() for c in native) >= cfg.min_text_chars_per_page:
            return native, False

    if not cfg.ocr_enabled:
        return "", False

    backend, obj = get_engine(cfg.ocr_backends)
    if backend is None:
        return "", False

    dpi = _render_dpi(page.rect, cfg, None)
    pix = page.get_pixmap(dpi=dpi)
    try:
        from PIL import Image  # noqa: F401
        import PIL.Image  # type: ignore

        PIL.Image.MAX_IMAGE_PIXELS = None
    except Exception:
        pass
    data = pix.tobytes("png")
    del pix
    text = _ocr_image(obj, backend, data)
    del data
    # ONNX runtime holds on to large intermediate buffers; releasing them
    # after every page keeps a 50-file batch from creeping into an OOM.
    import gc

    gc.collect()
    return text, True


def _worker_init(backends: tuple[str, ...]) -> None:
    """Pre-build the OCR engine inside each pool worker (once per process)."""
    get_engine(backends)


def _extract_one(path_str: str, cfg_dict: dict) -> dict:
    """Top-level function so it can be pickled into a ProcessPoolExecutor."""
    import pymupdf

    cfg = ExtractConfig(**cfg_dict)
    path = Path(path_str)
    res: dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "pages": 0,
        "text": "",
        "ocr_used": False,
        "error": None,
        "encrypted": False,
    }
    try:
        doc = pymupdf.open(path)
        if doc.needs_pass:
            res["encrypted"] = True
            res["error"] = "PDF is password protected"
            doc.close()
            return res
        res["pages"] = doc.page_count
        parts: list[str] = []
        ocr_any = False
        for i in range(doc.page_count):
            try:
                t, used = _page_text(doc, i, cfg, cfg.ocr_force)
            except Exception as exc:  # keep going: one bad page != dead resume
                log.warning("%s page %d failed: %s", path.name, i + 1, exc)
                t, used = "", False
            parts.append(t)
            ocr_any = ocr_any or used
        doc.close()

        text = "\n".join(parts)
        text = text.replace("\u00a0", " ").replace("\ufb01", "fi").replace("\ufb02", "fl")
        text = _MULTI_NL.sub("\n\n", _WS.sub(" ", text))
        res["text"] = text.strip()
        res["ocr_used"] = ocr_any
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
@dataclass
class ExtractedDoc:
    file: str
    path: str
    pages: int
    text: str
    ocr_used: bool
    error: str | None
    cached: bool = False
    sha1: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(cfg: ExtractConfig, sha: str) -> Path:
    d = Path(cfg.cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sha}.json"


def extract_pdf(path: Path, cfg: ExtractConfig) -> ExtractedDoc:
    sha = _sha1_of(path)
    if cfg.use_cache:
        cp = _cache_path(cfg, sha)
        if cp.exists():
            try:
                data = json.loads(cp.read_text())
                if data.get("text"):
                    doc = ExtractedDoc(**{**data, "cached": True, "sha1": sha})
                    doc.file = path.name  # file may have been renamed
                    return doc
            except Exception:
                pass
    raw = _extract_one(str(path), asdict(cfg))
    doc = ExtractedDoc(
        file=path.name,
        path=str(path),
        pages=raw["pages"],
        text=raw["text"],
        ocr_used=raw["ocr_used"],
        error=raw["error"],
        sha1=sha,
    )
    if cfg.use_cache and doc.text:
        try:
            _cache_path(cfg, sha).write_text(
                json.dumps(
                    {
                        "file": doc.file,
                        "path": doc.path,
                        "pages": doc.pages,
                        "text": doc.text,
                        "ocr_used": doc.ocr_used,
                        "error": doc.error,
                    }
                )
            )
        except Exception as exc:
            log.debug("cache write failed: %s", exc)
    return doc


def _total_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 8.0


def _auto_workers(cfg: ExtractConfig) -> int:
    """Pick an OCR worker count that fits in the RAM actually free right now.

    Measured ONNX peak per worker: ~0.7 GB at 1100px, ~1.2 GB at 1700px.
    A cold run never overlaps OCR with the LLM (extraction finishes and the
    pool is torn down before Ollama is contacted), but with keep_alive=-1 a
    model left resident by a previous run *does* overlap -- so budget against
    free memory, not total memory.
    """
    if cfg.ocr_workers:
        return max(1, cfg.ocr_workers)
    cpus = os.cpu_count() or 2
    try:
        import psutil  # type: ignore

        avail_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        avail_gb = _total_ram_gb()
    budget = int((avail_gb - 2.0) // 1.5)  # keep ~2 GB headroom
    return max(1, min(cpus, 4, budget))


def _extract_isolated(paths: list[Path], cfg: ExtractConfig) -> list[dict]:
    """One subprocess per file. Slower startup, but memory is fully reclaimed.

    This is the safe path for machines with < 6 GB RAM.
    """
    import json as _json
    import subprocess
    import sys

    out: list[dict] = []
    cfg_json = _json.dumps(asdict(cfg))
    for p in paths:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "resume_ranker.ocr_worker", str(p), cfg_json],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
                timeout=600,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                raise RuntimeError((proc.stderr or "no output").strip()[-300:])
            out.append(_json.loads(proc.stdout))
        except Exception as exc:
            log.warning("Isolated OCR failed for %s: %s", p.name, exc)
            out.append({"file": p.name, "path": str(p), "pages": 0, "text": "",
                        "ocr_used": False, "error": f"{type(exc).__name__}: {exc}"})
    return out


def extract_all(paths: list[Path], cfg: ExtractConfig) -> list[ExtractedDoc]:
    """Extract every PDF. Cache hits are free; the rest go to a process pool."""
    docs: list[ExtractedDoc] = []
    todo: list[Path] = []
    for p in paths:
        if cfg.use_cache:
            try:
                sha = _sha1_of(p)
                cp = _cache_path(cfg, sha)
                if cp.exists():
                    data = json.loads(cp.read_text())
                    if data.get("text"):
                        d = ExtractedDoc(**{**data, "cached": True, "sha1": sha})
                        d.file = p.name
                        docs.append(d)
                        continue
            except Exception:
                pass
        todo.append(p)

    if todo:
        # Strategy selection: a persistent pool is fastest, but each worker
        # holds ~1.2 GB of ONNX arena *and* a forked child inherits the
        # parent's pages. On a small machine that OOMs, so there we run one
        # short-lived subprocess per file instead (memory fully reclaimed).
        if _total_ram_gb() < 6.0 and len(todo) > 1:
            log.info("Extracting %d PDF(s) — low-RAM mode, one isolated "
                     "OCR process per file (slower but memory-safe).", len(todo))
            raws = _extract_isolated(todo, cfg)
        else:
            workers = _auto_workers(cfg)
            log.info("Extracting %d PDF(s) with %d persistent OCR worker(s)...",
                     len(todo), workers)
            cfg_d = asdict(cfg)
            # max_tasks_per_child=1 recycles workers so ONNX arenas are freed.
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(cfg.ocr_backends,),
                max_tasks_per_child=1,
            ) as pool:
                raws = list(pool.map(_extract_one, [str(p) for p in todo],
                                     [cfg_d] * len(todo)))
        for raw in raws:
            d = ExtractedDoc(
                file=Path(raw["path"]).name,
                path=raw["path"],
                pages=raw["pages"],
                text=raw["text"],
                ocr_used=raw["ocr_used"],
                error=raw["error"],
            )
            docs.append(d)
            try:
                sha = _sha1_of(Path(d.path))
                d.sha1 = sha
            except Exception:
                sha = ""
            if cfg.use_cache and d.text and sha:
                try:
                    _cache_path(cfg, sha).write_text(json.dumps({
                        "file": d.file, "path": d.path, "pages": d.pages,
                        "text": d.text, "ocr_used": d.ocr_used, "error": d.error,
                    }))
                except Exception:
                    pass
    order = {str(p): i for i, p in enumerate(paths)}
    docs.sort(key=lambda d: order.get(d.path, 1 << 30))
    return docs
