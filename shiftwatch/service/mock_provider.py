"""Ollama-compatible mock model server with controllable latency and faults.

It speaks POST /api/generate so the *real* OllamaProvider code path is exercised.
Faults are injected per request, deterministically from (seed, prompt, n-th call of
that prompt), so experiments are repeatable:

  --latency-ms / --jitter-ms   response time
  --error-rate                 HTTP 500
  --drop-rate                  close the socket without replying (connection error)
  --invalid-json-rate          HTTP 200 but model text violates the JSON contract
  --max-concurrency N          HTTP 429 when more than N requests are in flight
  --dataset PATH               answer with a required term (prob --accuracy) so that
                               scoring produces non-trivial numbers

GET /stats reports calls per prompt, so duplicate external calls can be counted.
This measures the *scheduler*, not LLM throughput; never report its numbers as such.
"""
import argparse
from collections import Counter
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import random
import re
import threading
import time


class MockState:
    def __init__(self, **config):
        self.config = config
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = Counter()
        self.outcomes = Counter()
        self.answers: dict[str, str] = {}

    def reset(self) -> None:
        with self.lock:
            self.max_in_flight = 0
            self.calls.clear()
            self.outcomes.clear()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "calls_total": sum(self.calls.values()),
                "distinct_prompts": len(self.calls),
                "prompts_called_more_than_once": sum(1 for n in self.calls.values() if n > 1),
                "extra_calls": sum(n - 1 for n in self.calls.values() if n > 1),
                "max_in_flight": self.max_in_flight,
                "outcomes": dict(self.outcomes),
                "config": self.config,
            }


def _load_answers(path: str) -> dict[str, str]:
    answers = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                answer = "I cannot know that." if item.get("should_abstain") else item["required_terms"][0]
                for prompt in item["prompts"].values():
                    answers[prompt] = answer
    return answers


def _user_prompt(full_prompt: str) -> str:
    match = re.search(r"User prompt:\n(.*)\Z", full_prompt, re.S)
    return match.group(1) if match else full_prompt


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence default stderr logging
            pass

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            try:
                self.send_response(code)
            except (BrokenPipeError, ConnectionResetError):
                return
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client (e.g. a killed worker) went away mid-request

        def do_GET(self):
            if self.path == "/stats":
                self._send(200, state.snapshot())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if self.path == "/reset":
                state.reset()
                self._send(200, {"ok": True})
                return
            if self.path != "/api/generate":
                self._send(404, {"error": "not found"})
                return
            prompt = _user_prompt(json.loads(raw)["prompt"])
            cfg = state.config
            outcome = "unknown"
            with state.lock:
                state.calls[prompt] += 1
                call_no = state.calls[prompt]
                state.in_flight += 1
                state.max_in_flight = max(state.max_in_flight, state.in_flight)
                over_limit = cfg["max_concurrency"] and state.in_flight > cfg["max_concurrency"]
            try:
                seed = hashlib.sha256(f"{cfg['seed']}|{prompt}|{call_no}".encode()).digest()
                rng = random.Random(seed)
                if over_limit:
                    outcome = "rate_limited"
                    self._send(429, {"error": "too many requests"})
                    return
                time.sleep(max(0.0, rng.gauss(cfg["latency_ms"], cfg["jitter_ms"])) / 1000)
                roll = rng.random()
                if roll < cfg["drop_rate"]:
                    outcome = "dropped"
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                roll -= cfg["drop_rate"]
                if roll < cfg["error_rate"]:
                    outcome = "server_error"
                    self._send(500, {"error": "injected failure"})
                    return
                roll -= cfg["error_rate"]
                if roll < cfg["invalid_json_rate"]:
                    outcome = "invalid_json"
                    text = "Sure! The answer is probably fine."
                else:
                    outcome = "ok"
                    correct = rng.random() < cfg["accuracy"]
                    answer = state.answers.get(prompt, "mock answer") if correct else "I am not sure, maybe something else."
                    text = json.dumps({
                        "answer": answer,
                        "confidence": round(rng.uniform(0.5, 1.0), 2),
                        "abstain": False,
                    })
                self._send(200, {"model": "mock", "response": text, "done": True,
                                 "prompt_eval_count": len(prompt.split()), "eval_count": len(text.split())})
            finally:
                with state.lock:
                    state.in_flight -= 1
                    state.outcomes[outcome] += 1

    return Handler


def serve(host="127.0.0.1", port=0, *, latency_ms=100.0, jitter_ms=20.0, error_rate=0.0,
          drop_rate=0.0, invalid_json_rate=0.0, max_concurrency=0, accuracy=0.8,
          seed=7, dataset=None):
    """Start in a background thread; returns (server, state). Use port=0 for a free port."""
    state = MockState(latency_ms=latency_ms, jitter_ms=jitter_ms, error_rate=error_rate,
                      drop_rate=drop_rate, invalid_json_rate=invalid_json_rate,
                      max_concurrency=max_concurrency, accuracy=accuracy, seed=seed)
    if dataset:
        state.answers = _load_answers(dataset)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11500)
    parser.add_argument("--latency-ms", type=float, default=100)
    parser.add_argument("--jitter-ms", type=float, default=20)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--invalid-json-rate", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=0)
    parser.add_argument("--accuracy", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset")
    args = parser.parse_args(argv)
    server, _ = serve(args.host, args.port, latency_ms=args.latency_ms, jitter_ms=args.jitter_ms,
                      error_rate=args.error_rate, drop_rate=args.drop_rate,
                      invalid_json_rate=args.invalid_json_rate, max_concurrency=args.max_concurrency,
                      accuracy=args.accuracy, seed=args.seed, dataset=args.dataset)
    print(f"mock provider listening on {args.host}:{server.server_address[1]}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
