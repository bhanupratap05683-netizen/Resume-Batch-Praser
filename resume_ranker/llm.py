"""Async client for a *local* Ollama server.

Efficiency notes that matter for a 7B model on one machine:
  * `format: "json"` forces constrained decoding -- far fewer malformed
    outputs than prompt-pleading, and it costs nothing extra.
  * Small `num_ctx` (4096 instead of the 32k default) dramatically cuts
    prefill/memory. A resume chunk does not need 32k.
  * `num_predict` caps generation so one runaway answer can't stall the batch.
  * `keep_alive=-1` keeps weights resident: saves ~2-4 s of model loading
    on every single resume.
  * A semaphore caps in-flight requests. Set it to your OLLAMA_NUM_PARALLEL.
  * Everything is retried with a slightly re-worded "JSON only" reminder,
    because small models occasionally drift into prose.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import OllamaConfig

log = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    data: Any
    seconds: float
    eval_tokens: int = 0
    tokens_per_second: float = 0.0


def extract_json(raw: str) -> Any:
    """Best-effort JSON parse that survives the usual small-model damage."""
    if not raw:
        raise LLMError("empty response")
    txt = raw.strip()
    txt = _JSON_FENCE.sub(lambda m: m.group(1), txt).strip()

    try:
        return json.loads(txt)
    except Exception:
        pass

    # Grab the outermost balanced {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = txt.find(opener)
        end = txt.rfind(closer)
        if start != -1 and end > start:
            frag = txt[start : end + 1]
            try:
                return json.loads(frag)
            except Exception:
                frag2 = _repair(frag)
                try:
                    return json.loads(frag2)
                except Exception:
                    continue
    raise LLMError(f"could not parse JSON from: {raw[:180]!r}")


def _repair(frag: str) -> str:
    frag = re.sub(r",\s*([\]}])", r"\1", frag)          # trailing commas
    frag = re.sub(r"[\u201c\u201d]", '"', frag)         # smart quotes
    frag = re.sub(r"\bTrue\b", "true", frag)
    frag = re.sub(r"\bFalse\b", "false", frag)
    frag = re.sub(r"\bNone\b", "null", frag)
    frag = re.sub(r"//[^\n\"]*", "", frag)              # strip // comments
    frag = re.sub(r"\s*\n\s*", " ", frag)
    return frag


class OllamaLLM:
    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(max(1, cfg.concurrency))
        self._client: httpx.AsyncClient | None = None
        self._calls = 0
        self._tokens = 0
        self._seconds = 0.0

    # -- lifecycle ---------------------------------------------------------
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.cfg.host.rstrip("/"),
            timeout=httpx.Timeout(self.cfg.timeout_s, connect=15.0),
            limits=httpx.Limits(max_connections=self.cfg.concurrency + 4,
                                max_keepalive_connections=self.cfg.concurrency + 4),
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "tokens": self._tokens,
            "seconds": round(self._seconds, 1),
            "tok_per_s": round(self._tokens / self._seconds, 1) if self._seconds else 0.0,
        }

    # -- server info -------------------------------------------------------
    async def health(self) -> dict[str, Any]:
        """Check the server is up and the model is actually pulled."""
        assert self._client
        try:
            r = await self._client.get("/api/tags")
            r.raise_for_status()
        except Exception as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.cfg.host}. "
                f"Start it with `ollama serve`. ({type(exc).__name__})"
            ) from exc
        tags = r.json().get("models", []) or []
        names = [m.get("name", "") for m in tags]
        have = any(
            n == self.cfg.model or n.split(":")[0] == self.cfg.model.split(":")[0]
            for n in names
        )
        return {"ok": True, "models": names, "model_present": have}

    async def warmup(self) -> None:
        if not self.cfg.warmup or not self._client:
            return
        try:
            await self.chat_json(
                system="Reply with JSON only.",
                prompt='Return exactly: {"ok": true}',
                num_ctx=512,
                num_predict=16,
            )
            log.info("Model %s warmed up and resident.", self.cfg.model)
        except Exception as exc:
            log.warning("Warmup failed (non-fatal): %s", exc)

    # -- core --------------------------------------------------------------
    def _options(self, num_ctx: int, num_predict: int) -> dict[str, Any]:
        o: dict[str, Any] = {
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "repeat_penalty": float(self.cfg.repeat_penalty),
        }
        if self.cfg.num_gpu is not None:
            o["num_gpu"] = int(self.cfg.num_gpu)
        if self.cfg.num_thread is not None:
            o["num_thread"] = int(self.cfg.num_thread)
        return o

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._client
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = await self._client.post("/api/chat", json=payload)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, httpx.StreamError) as exc:
                last = exc
                wait = 1.5 * (attempt + 1)
                log.debug("Ollama call failed (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(wait)
        raise LLMError(f"Ollama request failed after retries: {last}")

    async def generate(
        self,
        prompt: str,
        system: str = "",
        num_ctx: int | None = None,
        num_predict: int | None = None,
        json_mode: bool = True,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}],
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": self._options(num_ctx or self.cfg.chunk_ctx,
                                     num_predict or self.cfg.chunk_predict),
        }
        if json_mode:
            payload["format"] = "json"

        t0 = time.perf_counter()
        async with self._sem:
            raw = await self._post(payload)
        elapsed = time.perf_counter() - t0

        content = (raw.get("message") or {}).get("content", "") or ""
        eval_count = int(raw.get("eval_count") or 0)
        eval_ns = int(raw.get("eval_duration") or 0)
        self._calls += 1
        self._tokens += eval_count
        self._seconds += elapsed
        return LLMResponse(
            text=content,
            data=extract_json(content) if json_mode else content,
            seconds=elapsed,
            eval_tokens=eval_count,
            tokens_per_second=(eval_count / (eval_ns / 1e9)) if eval_ns else 0.0,
        )

    async def chat_json(
        self,
        prompt: str,
        system: str = "",
        num_ctx: int | None = None,
        num_predict: int | None = None,
        repair_hint: str = "Respond with a single valid JSON object only. No prose.",
    ) -> Any:
        """Generate and parse JSON, retrying once with a stiffer reminder."""
        try:
            resp = await self.generate(prompt, system, num_ctx, num_predict, json_mode=True)
            return resp.data
        except LLMError:
            resp = await self.generate(
                prompt + "\n\n" + repair_hint, system, num_ctx, num_predict, json_mode=True
            )
            return resp.data
