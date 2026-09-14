"""A fake Ollama server, used to test the pipeline without a real model.

Implements just enough of the API:
    GET  /api/tags          -> pretend qwen2.5-coder:7b is pulled
    POST /api/chat          -> return a JSON "judgement" for the prompt

It derives its score from how many JD keywords appear in the prompt, so the
numbers move the way a real model's would. Handy for exercising the async
client, retries, JSON repair and the score blend.

    python tools/mock_ollama.py --port 11555
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "qwen2.5-coder:7b"

KEYWORDS = [
    "python", "pandas", "numpy", "scipy", "statsmodels", "sql", "window function",
    "cte", "indexing", "excel", "power query", "vba", "macro", "pivot",
    "arima", "garch", "var", "sharpe", "time series", "regression",
    "probability", "hypothesis", "monte carlo", "backtest", "dcf", "valuation",
    "derivatives", "fixed income", "portfolio", "power bi", "tableau",
    "streamlit", "machine learning", "random forest", "clustering",
    "snowflake", "bigquery", "git", "github", "cfa", "frm", "bloomberg",
    "factset", "dashboard", "etl", "forecast", "risk",
]


def fake_judgement(prompt: str) -> dict:
    low = prompt.lower()
    hits = [k for k in KEYWORDS if k in low]
    n = len(hits)
    base = min(96, 12 + n * 3.1)
    score = int(max(3, min(97, base + random.uniform(-5, 5))))
    if score >= 75:
        rec = "strong_fit"
    elif score >= 58:
        rec = "moderate_fit"
    else:
        rec = "weak_fit"
    strengths = [f"Evidence of {h} in the resume" for h in hits[:3]] or [
        "No clearly relevant experience surfaced"]
    gaps = [f"No mention of {k}" for k in ("ARIMA/GARCH", "VaR", "Power BI") if k.lower() not in low]
    return {
        "score": score,
        "confidence": round(random.uniform(0.5, 0.95), 2),
        "years_relevant": min(12, max(0, n // 3)),
        "seniority": ["intern", "junior", "mid", "senior"][min(3, n // 5)],
        "strengths": strengths[:3],
        "gaps": gaps[:3],
        "reasoning": (
            f"Resume matches {n} of the role's key terms ({', '.join(hits[:5]) or 'none'}). "
            f"Depth appears {rec.replace('_', ' ')}; evidence is drawn from the "
            f"experience and skills sections."
        ),
        "recommendation": rec,
    }


ROOT_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Mock Ollama — running</title>
<style>
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
     max-width:760px;margin:48px auto;padding:0 20px;color:#111;background:#fafafa}
h1{font-size:22px;margin-bottom:4px}
.box{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;
     margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
code{background:#f3f4f6;padding:2px 6px;border-radius:4px;font-size:13px}
.ok{color:#16a34a;font-weight:600}
table{width:100%;border-collapse:collapse;margin-top:8px}
td{padding:6px 0;border-top:1px solid #f3f4f6;font-size:14px}
td:first-child{color:#6b7280;width:150px}
a{color:#2563eb}
</style></head><body>
<h1>✅ Mock Ollama is running</h1>
<p class="ok">Stand-in for your local qwen2.5-coder:7b — no GPU needed.</p>
<div class="box">
<table>
<tr><td>Endpoint</td><td><code>http://0.0.0.0:__PORT__</code></td></tr>
<tr><td>Model</td><td><code>__MODEL__</code></td></tr>
<tr><td>Chat requests</td><td><code>POST /api/chat</code> — <b>__NREQ__</b> served so far</td></tr>
<tr><td>Model list</td><td><code>GET /api/tags</code></td></tr>
</table></div>
<div class="box">
<p>Run the real pipeline against this stub from a terminal in the sandbox:</p>
<p><code>python main.py run --pdf-dir uploads --host http://127.0.0.1:__PORT__ --concurrency 4</code></p>
<p>It invents plausible scores so you can exercise batching, JSON parsing and
the score blend. Replace the host with <code>http://127.0.0.1:11434</code> to use
your real Ollama model.</p>
</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    n_requests = 0
    port = 11555

    def log_message(self, *args):  # keep the console quiet
        pass

    def _send_html(self, html: str, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._send({"models": [{"name": MODEL, "size": 4_000_000_000}]})
        elif self.path == "/" or self.path.startswith("/?"):
            self._send_html(ROOT_PAGE.replace("__NREQ__", str(Handler.n_requests))
                            .replace("__MODEL__", MODEL)
                            .replace("__PORT__", str(Handler.port)))
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/api/chat"):
            self._send({"error": "not found"}, 404)
            return
        Handler.n_requests += 1
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        # Simulate real inference latency so concurrency is actually tested.
        time.sleep(random.uniform(0.25, 0.7))
        prompt = " ".join(m.get("content", "") for m in payload.get("messages", []))
        content = json.dumps(fake_judgement(prompt))
        n_tokens = max(1, len(content) // 4)
        self._send({
            "model": payload.get("model", MODEL),
            "message": {"role": "assistant", "content": content},
            "eval_count": n_tokens,
            "eval_duration": int(n_tokens / 28 * 1e9),  # pretend ~28 tok/s
            "done": True,
        })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11555)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    Handler.port = args.port
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock Ollama listening on http://{args.host}:{args.port}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
