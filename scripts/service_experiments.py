"""Reliability and scheduling experiments for the evaluation service.

Runs real worker *processes* (so SIGKILL is a real crash) against PostgreSQL and the
fault-injecting mock provider. Results measure the scheduler, not LLM inference.

    SHIFTWATCH_DATABASE_URL=postgresql://postgres@localhost/shiftwatch_exp \
      python scripts/service_experiments.py --output docs/service_experiments.json

The target database is wiped. Never point this at a database you care about.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shiftwatch.service import repository  # noqa: E402
from shiftwatch.service.config import Settings  # noqa: E402
from shiftwatch.service.db import connect, migrate  # noqa: E402

DATASET = ROOT / "datasets" / "llm_benchmark_60.jsonl"
MOCK_PORT = 11561


def load_cases():
    return [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]


class Env:
    def __init__(self, db_url):
        self.db_url = db_url
        self.procs = []

    def reset_db(self):
        migrate(self.db_url)
        with connect(self.db_url) as conn:
            conn.execute(
                "TRUNCATE scores, scoring_runs, model_outputs, task_attempts, inference_tasks,"
                " inference_runs, provider_limits, model_configs, dataset_cases, dataset_versions"
            )

    def start_mock(self, **kw):
        args = [sys.executable, "-m", "shiftwatch.service", "mock-provider", "--host", "127.0.0.1",
                "--port", str(MOCK_PORT), "--dataset", str(DATASET)]
        for key, value in kw.items():
            args += [f"--{key.replace('_', '-')}", str(value)]
        proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, text=True)
        proc.stdout.readline()  # "listening" line
        self.procs.append(proc)
        return proc

    def start_worker(self, name, slots=4, **env):
        full_env = {**os.environ, "SHIFTWATCH_DATABASE_URL": self.db_url,
                    "SHIFTWATCH_RETRY_BASE_SECONDS": "0.2", "SHIFTWATCH_RETRY_MAX_SECONDS": "2",
                    **{k: str(v) for k, v in env.items()}}
        proc = subprocess.Popen(
            [sys.executable, "-m", "shiftwatch.service", "worker", "--slots", str(slots),
             "--worker-id", name],
            cwd=ROOT, env=full_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.procs.append(proc)
        return proc

    def stop_all(self):
        for proc in self.procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in self.procs:
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.procs = []

    def mock_stats(self):
        from urllib import request
        with request.urlopen(f"http://127.0.0.1:{MOCK_PORT}/stats") as r:
            return json.load(r)

    def submit(self, conn, name, cases, **kw):
        dataset = repository.create_dataset_version(conn, name, cases)
        params = {"model": "mock", "base_url": f"http://127.0.0.1:{MOCK_PORT}", "timeout": 10}
        run, _ = repository.submit_run(conn, repository.RunRequest(
            dataset_version_id=dataset["id"], provider="ollama", params=params, **kw
        ), replace(Settings.from_env(), database_url=self.db_url))
        return run

    def wait_run(self, conn, run_id, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = conn.execute("SELECT status FROM inference_runs WHERE id = %s", (run_id,)).fetchone()["status"]
            conn.commit()
            if status not in ("queued", "running"):
                return status
            time.sleep(0.1)
        raise TimeoutError(run_id)


def run_metrics(conn, run_id):
    row = conn.execute(
        "SELECT r.status, extract(epoch FROM r.finished_at - r.created_at) AS makespan,"
        " (SELECT count(*) FROM inference_tasks WHERE run_id = r.id AND status NOT IN"
        "   ('succeeded','failed','cancelled')) AS lost,"
        " (SELECT count(*) FROM inference_tasks WHERE run_id = r.id AND status = 'succeeded') AS succeeded,"
        " (SELECT count(*) FROM inference_tasks WHERE run_id = r.id AND status = 'failed') AS failed"
        " FROM inference_runs r WHERE r.id = %s", (run_id,)).fetchone()
    attempts = conn.execute(
        "SELECT a.outcome, count(*) AS n FROM task_attempts a JOIN inference_tasks t ON t.id = a.task_id"
        " WHERE t.run_id = %s GROUP BY a.outcome", (run_id,)).fetchall()
    dup = conn.execute(
        "SELECT count(*) AS n FROM (SELECT a.task_id FROM task_attempts a JOIN inference_tasks t"
        " ON t.id = a.task_id WHERE t.run_id = %s AND a.outcome = 'succeeded'"
        " GROUP BY a.task_id HAVING count(*) > 1) x", (run_id,)).fetchone()["n"]
    waits = [r["w"] for r in conn.execute(
        "SELECT extract(epoch FROM min(a.claimed_at) - t.created_at) AS w FROM task_attempts a"
        " JOIN inference_tasks t ON t.id = a.task_id WHERE t.run_id = %s GROUP BY t.id, t.created_at",
        (run_id,)).fetchall()]
    waits.sort()
    return {
        "status": row["status"], "makespan_s": round(float(row["makespan"]), 2),
        "succeeded": row["succeeded"], "failed": row["failed"], "lost_tasks": row["lost"],
        "duplicate_effective_results": dup,
        "attempt_outcomes": {r["outcome"] or "unfinished": r["n"] for r in attempts},
        "queue_wait_p50_s": round(float(waits[len(waits) // 2]), 2),
        "queue_wait_p95_s": round(float(waits[int(len(waits) * 0.95) - 1]), 2),
    }


def xact_commits(conn):
    value = conn.execute(
        "SELECT xact_commit FROM pg_stat_database WHERE datname = current_database()").fetchone()["xact_commit"]
    conn.commit()
    return value


# --------------------------------------------------------------------------- experiments

def crash_recovery(env, trials=3):
    results = []
    for trial in range(trials):
        env.reset_db()
        env.start_mock(latency_ms=300, jitter_ms=60, seed=100 + trial)
        with connect(env.db_url) as conn:
            run = env.submit(conn, "bench", load_cases())
            conn.commit()
            repository.set_provider_limit(conn, f"ollama:http://127.0.0.1:{MOCK_PORT}", 8)
            victim = env.start_worker("victim", slots=4, SHIFTWATCH_LEASE_SECONDS=3)
            env.start_worker("survivor", slots=4, SHIFTWATCH_LEASE_SECONDS=3)
            while conn.execute("SELECT count(*) AS n FROM model_outputs").fetchone()["n"] < 96:
                conn.commit()
                time.sleep(0.05)
            conn.commit()
            victim.kill()  # SIGKILL: no cleanup, leases left dangling
            killed_at = time.time()
            orphaned = conn.execute(
                "SELECT count(*) AS n FROM inference_tasks WHERE status = 'running' AND lease_owner = 'victim'"
            ).fetchone()["n"]
            conn.commit()
            env.start_worker("replacement", slots=4, SHIFTWATCH_LEASE_SECONDS=3)
            # Recovery = until every task the victim held has a committed output again.
            victim_tasks = [r["task_id"] for r in conn.execute(
                "SELECT task_id FROM task_attempts WHERE worker_id = 'victim' AND outcome IS NULL").fetchall()]
            conn.commit()
            while True:
                done = conn.execute(
                    "SELECT count(*) AS n FROM inference_tasks WHERE id = ANY(%s) AND status = 'succeeded'",
                    (victim_tasks,)).fetchone()["n"]
                conn.commit()
                if done == len(victim_tasks):
                    break
                time.sleep(0.05)
            recovery_s = time.time() - killed_at
            env.wait_run(conn, run["id"])
            metrics = run_metrics(conn, run["id"])
        mock = env.mock_stats()
        env.stop_all()
        results.append({
            "trial": trial, "tasks": 240, "orphaned_running_at_kill": orphaned,
            "recovery_s": round(recovery_s, 2), **metrics,
            "provider_calls": mock["calls_total"], "extra_provider_calls": mock["extra_calls"],
        })
        print("crash", results[-1], flush=True)
    return {"setup": "240 tasks, mock 300±60 ms, provider cap 8, 2 workers x 4 slots, lease 3 s; "
                     "SIGKILL one worker after 96 outputs, start a replacement immediately",
            "trials": results}


def scaling(env):
    results = []
    for workers in (1, 2, 4):
        env.reset_db()
        env.start_mock(latency_ms=200, jitter_ms=40, error_rate=0.03)
        with connect(env.db_url) as conn:
            repository.set_provider_limit(conn, f"ollama:http://127.0.0.1:{MOCK_PORT}", 64)
            run = env.submit(conn, "bench", load_cases())
            conn.commit()
            commits_before = xact_commits(conn)
            for i in range(workers):
                env.start_worker(f"w{i}", slots=4)
            env.wait_run(conn, run["id"])
            commits = xact_commits(conn) - commits_before
            metrics = run_metrics(conn, run["id"])
        mock = env.mock_stats()
        env.stop_all()
        attempts = sum(metrics["attempt_outcomes"].values())
        results.append({
            "workers": workers, "slots_total": workers * 4, **metrics,
            "throughput_tasks_per_s": round(240 / metrics["makespan_s"], 1),
            "db_commits_per_task": round(commits / 240, 1),
            "attempt_error_rate": round(1 - metrics["attempt_outcomes"].get("succeeded", 0) / attempts, 3),
            "max_in_flight_at_provider": mock["max_in_flight"],
        })
        print("scaling", results[-1], flush=True)
    return {"setup": "240 tasks, mock 200±40 ms with 3% HTTP 500, provider cap 64 (not binding), "
                     "N workers x 4 slots", "results": results}


def fairness(env):
    results = []
    variants = [("fifo", 4), ("fair", 4), ("fair", 16)]
    for scheduler, cap in variants:
        env.reset_db()
        env.start_mock(latency_ms=200, jitter_ms=40, max_concurrency=4)
        with connect(env.db_url) as conn:
            repository.set_provider_limit(conn, f"ollama:http://127.0.0.1:{MOCK_PORT}", cap)
            big = env.submit(conn, "big", load_cases(), max_attempts=10)
            conn.commit()
            env.start_worker("w0", slots=8, SHIFTWATCH_SCHEDULER=scheduler)
            time.sleep(1.0)
            small = env.submit(conn, "small", load_cases()[:4], max_attempts=10)
            conn.commit()
            env.wait_run(conn, small["id"])
            env.wait_run(conn, big["id"])
            small_m, big_m = run_metrics(conn, small["id"]), run_metrics(conn, big["id"])
        mock = env.mock_stats()
        env.stop_all()
        results.append({
            "scheduler": scheduler, "provider_cap": cap,
            "small_run_makespan_s": small_m["makespan_s"], "small_queue_wait_p95_s": small_m["queue_wait_p95_s"],
            "big_run_makespan_s": big_m["makespan_s"],
            "provider_429s": mock["outcomes"].get("rate_limited", 0),
            "retryable_attempts": big_m["attempt_outcomes"].get("retryable_error", 0)
            + small_m["attempt_outcomes"].get("retryable_error", 0),
            "max_in_flight_at_provider": mock["max_in_flight"],
            "failed_tasks": big_m["failed"] + small_m["failed"],
        })
        print("fairness", results[-1], flush=True)
    return {"setup": "big run (240 tasks) then small run (16 tasks) 1 s later; 1 worker x 8 slots; "
                     "mock returns 429 above 4 concurrent requests", "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/service_experiments.json")
    parser.add_argument("--only", choices=("crash", "scaling", "fairness"))
    args = parser.parse_args()
    db_url = os.environ["SHIFTWATCH_DATABASE_URL"]
    env = Env(db_url)
    report = {}
    try:
        if args.only in (None, "crash"):
            report["crash_recovery"] = crash_recovery(env)
        if args.only in (None, "scaling"):
            report["scaling"] = scaling(env)
        if args.only in (None, "fairness"):
            report["fairness"] = fairness(env)
    finally:
        env.stop_all()
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
