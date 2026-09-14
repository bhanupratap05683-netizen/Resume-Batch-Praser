"""Configuration dataclasses for the resume ranking pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class OllamaConfig:
    """Tuning knobs for a *local* Ollama server.

    Every default here is chosen for a single-GPU / Apple-Silicon / CPU
    workstation running qwen2.5-coder:7b.
    """

    host: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5-coder:7b"
    # How many requests Ollama may work on *concurrently*.
    #   Keep this <= OLLAMA_NUM_PARALLEL you started the server with.
    concurrency: int = 4
    # Context windows. Small ctx = much faster prefill for a 7B model.
    chunk_ctx: int = 4096        # per-chunk scoring call
    reduce_ctx: int = 8192       # final aggregation call
    chunk_predict: int = 320     # cap output tokens -> latency control
    reduce_predict: int = 420
    temperature: float = 0.0     # deterministic scoring
    top_p: float = 0.9
    repeat_penalty: float = 1.05
    # -1 keeps the model resident between runs (huge win across 50 files).
    keep_alive: str = "-1m"
    timeout_s: float = 180.0
    max_retries: int = 2
    num_gpu: int | None = None      # None = let Ollama decide
    num_thread: int | None = None   # None = let Ollama decide
    # Warm the model up with a tiny call so the first real resume isn't slow.
    warmup: bool = True


@dataclass
class ExtractConfig:
    """PDF -> text settings (including the OCR fallback for scanned PDFs)."""

    # Text-layer PDFs: use them directly. Below this many alphabetic chars
    # per page we assume the page is a scan and fall back to OCR.
    min_text_chars_per_page: int = 120
    ocr_enabled: bool = True
    # Backends tried in order: "rapidocr" (pure pip), "tesseract" (system bin),
    # "paddleocr" (heavy). First one that imports wins.
    ocr_backends: tuple[str, ...] = ("rapidocr", "tesseract", "paddleocr")
    ocr_force: bool = False          # True = always OCR (ignore text layer)
    # Render the page so its LONG side is ~this many pixels, then clamp DPI.
    #   1700px is the sweet spot: quality stays high, memory stays ~1GB.
    #   Raise to 2200-2600 on a 16GB+ machine for dense two-column resumes.
    ocr_target_long_side: int = 1700
    ocr_min_dpi: int = 200
    ocr_max_dpi: int = 400
    ocr_dpi: int | None = None       # hard override
    # CPU-bound OCR runs in a process pool (escapes the GIL).
    ocr_workers: int = 0             # 0 = auto (min(cpus, 4), capped by RAM)
    cache_dir: str = "cache"
    use_cache: bool = True           # never OCR the same file twice


@dataclass
class ChunkConfig:
    """Section-aware chunking so a 7B model never chokes on a long CV."""

    max_chars: int = 2400            # ~600 tokens
    overlap_chars: int = 250
    # Only the most JD-relevant chunks are sent to the LLM.
    max_chunks_to_llm: int = 6
    min_chunk_chars: int = 60
    # If the kept chunks total less than this many characters (~2.2k tokens)
    # we judge them in ONE call; otherwise fall back to map-reduce.
    single_pass_max_chars: int = 9000


@dataclass
class ScoreConfig:
    """How the deterministic rubric and the LLM judgement are blended."""

    weight_deterministic: float = 0.40
    weight_llm: float = 0.60
    # Deterministic scores cluster low because keyword coverage is sparse.
    # Raised from 1.35 -> 1.7 after matching was made precise: short aliases
    # no longer fire inside ordinary words ("arch" in "research"), so raw
    # coverage dropped ~45%. Tune this if your rubric is denser/sparser.
    deterministic_gain: float = 1.7
    deterministic_cap: float = 100.0
    # Flag candidates where the two scorers strongly disagree.
    disagreement_threshold: float = 25.0
    # Recommendation cut-offs (on the final 0..100 score).
    strong_fit: float = 75.0
    moderate_fit: float = 58.0
    # Penalty applied when a hard requirement from the rubric is missing.
    missing_must_have_penalty: float = 6.0
    # Penalty when the highest degree is unrelated to the role's fields.
    unrelated_degree_penalty: float = 5.0
    # Bonus for each year of relevant experience, saturating at this cap.
    experience_bonus_per_year: float = 1.2
    experience_bonus_cap_years: int = 8


@dataclass
class Config:
    """Top-level config object threaded through the whole pipeline."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    top_n: int = 10
    use_llm: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
