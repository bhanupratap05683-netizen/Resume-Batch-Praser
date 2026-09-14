"""Standalone OCR worker: one PDF, one process, then exit.

    python -m resume_ranker.ocr_worker <pdf_path> <config_json>

Used when the machine is memory-tight (< ~6 GB). ONNX runtime grabs a large
arena and never returns it, so a long-lived worker slowly starves the system.
Running one process per file guarantees the OS reclaims everything between
files, at the cost of ~2 s of interpreter + model startup per resume.
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: ocr_worker.py <pdf> <config-json>", file=sys.stderr)
        return 2
    from .config import ExtractConfig
    from .extract import _extract_one

    cfg = ExtractConfig(**json.loads(argv[2]))
    result = _extract_one(argv[1], json.loads(argv[2]))
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
