"""Tiny stdlib HTTP client for the service API.

  python -m shiftwatch.service client submit datasets/llm_benchmark_60.jsonl \
      --model llama3:latest [--base-url http://localhost:11434] [--wait]
  python -m shiftwatch.service client status RUN_ID
  python -m shiftwatch.service client score RUN_ID [--threshold 0.8] [--overrides overrides.json]
  python -m shiftwatch.service client cancel RUN_ID
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib import error, request


def call(api: str, method: str, path: str, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = request.Request(
        api.rstrip("/") + path, data=data, method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except error.HTTPError as http_error:
        sys.exit(f"{method} {path} -> {http_error.code}: {http_error.read().decode()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=os.environ.get("SHIFTWATCH_API", "http://localhost:8000"))
    sub = parser.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("dataset")
    submit.add_argument("--name")
    submit.add_argument("--provider", default="ollama", choices=("ollama", "fixture"))
    submit.add_argument("--model")
    submit.add_argument("--base-url", default="http://localhost:11434")
    submit.add_argument("--response-mode", default="short", choices=("short", "free"))
    submit.add_argument("--label")
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--max-cases", type=int)
    submit.add_argument("--idempotency-key")
    submit.add_argument("--wait", action="store_true")
    for name in ("status", "cancel"):
        sub.add_parser(name).add_argument("run_id")
    score = sub.add_parser("score")
    score.add_argument("run_id")
    score.add_argument("--threshold", type=float, default=0.8)
    score.add_argument("--overrides", help="JSON file: {case_id: {required_terms: [...]}}")
    args = parser.parse_args(argv)

    if args.command == "submit":
        path = Path(args.dataset)
        cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        dataset = call(args.api, "POST", "/datasets", {"name": args.name or path.stem, "cases": cases})
        params = {}
        if args.provider == "ollama":
            params = {"model": args.model, "base_url": args.base_url,
                      "response_mode": args.response_mode, "temperature": 0, "seed": 7}
        headers = {"Idempotency-Key": args.idempotency_key} if args.idempotency_key else {}
        run = call(args.api, "POST", "/runs", {
            "dataset_version_id": dataset["id"], "provider": args.provider, "params": params,
            "label": args.label, "priority": args.priority, "max_cases": args.max_cases,
        }, headers)
        print(json.dumps({"dataset_version_id": dataset["id"], "run_id": run["id"]}))
        while args.wait:
            status = call(args.api, "GET", f"/runs/{run['id']}")
            print(f"\r{status['status']:<16} {status['progress']:.0%} {status['task_counts']}", end="", file=sys.stderr)
            if status["status"] not in ("queued", "running"):
                print(file=sys.stderr)
                break
            time.sleep(2)
    elif args.command == "status":
        print(json.dumps(call(args.api, "GET", f"/runs/{args.run_id}"), indent=2, default=str))
    elif args.command == "cancel":
        print(json.dumps(call(args.api, "POST", f"/runs/{args.run_id}/cancel")))
    elif args.command == "score":
        overrides = json.loads(Path(args.overrides).read_text()) if args.overrides else {}
        scoring = call(args.api, "POST", f"/runs/{args.run_id}/scorings",
                       {"confidence_threshold": args.threshold, "case_overrides": overrides})
        print(json.dumps(call(args.api, "GET", f"/scorings/{scoring['id']}"), indent=2, default=str))


if __name__ == "__main__":
    main()
