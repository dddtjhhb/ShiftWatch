"""Integration tests for the evaluation service.

They need PostgreSQL and the service extras. Set SHIFTWATCH_TEST_DATABASE_URL to a
throwaway database (all tables are truncated between tests); otherwise they skip.
"""
import json
import os
import threading
import time
import unittest
from pathlib import Path

DB_URL = os.environ.get("SHIFTWATCH_TEST_DATABASE_URL")
try:
    import psycopg  # noqa: F401
    import fastapi  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

ROOT = Path(__file__).resolve().parents[1]

if HAVE_DEPS:
    from fastapi.testclient import TestClient

    from shiftwatch.service import queue, repository, scoring
    from shiftwatch.service.api import create_app
    from shiftwatch.service.config import Settings
    from shiftwatch.service.db import connect, migrate
    from shiftwatch.service.mock_provider import serve as serve_mock
    from shiftwatch.service.worker import Worker


def load_jsonl(name):
    return [json.loads(l) for l in (ROOT / "datasets" / name).read_text().splitlines() if l.strip()]


def settings(**overrides):
    base = dict(
        database_url=DB_URL, max_pending_tasks=10_000, max_tasks_per_run=5_000,
        default_max_attempts=3, default_provider_concurrency=4, lease_seconds=30,
        retry_base_seconds=0.0, retry_max_seconds=0.0, scheduler="fair",
    )
    base.update(overrides)
    return Settings(**base)


@unittest.skipUnless(HAVE_DEPS and DB_URL, "needs service extras and SHIFTWATCH_TEST_DATABASE_URL")
class ServiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        migrate(DB_URL)

    def setUp(self):
        self.conn = connect(DB_URL)
        self.conn.execute(
            "TRUNCATE scores, scoring_runs, model_outputs, task_attempts, inference_tasks,"
            " inference_runs, provider_limits, model_configs, dataset_cases, dataset_versions"
        )
        self.conn.commit()
        self.settings = settings()

    def tearDown(self):
        self.conn.close()

    def submit(self, cases=None, provider="fixture", params=None, **kwargs):
        cases = cases if cases is not None else load_jsonl("llm_demo.jsonl")
        dataset = repository.create_dataset_version(self.conn, "demo", cases)
        run, _ = repository.submit_run(self.conn, repository.RunRequest(
            dataset_version_id=dataset["id"], provider=provider, params=params or {}, **kwargs
        ), self.settings)
        return run

    def claim(self, worker="w1", lease_seconds=30, conn=None):
        return queue.claim_task(conn or self.conn, worker, lease_seconds, 4, "fair")

    def output(self, answer="mercury"):
        return {"raw_text": "{}", "answer": answer, "confidence": 0.9,
                "abstain": False, "latency_ms": 5}


class DataModelTests(ServiceTestCase):
    def test_dataset_versions_are_content_addressed(self):
        cases = load_jsonl("llm_demo.jsonl")
        first = repository.create_dataset_version(self.conn, "demo", cases)
        again = repository.create_dataset_version(self.conn, "demo", cases)
        changed = repository.create_dataset_version(self.conn, "demo", cases[:-1])
        self.assertTrue(first["created"])
        self.assertFalse(again["created"])
        self.assertEqual(first["id"], again["id"])
        self.assertNotEqual(first["id"], changed["id"])

    def test_invalid_dataset_rejected(self):
        with self.assertRaises(ValueError):
            repository.create_dataset_version(self.conn, "bad", [{"id": "x"}])

    def test_run_expands_cases_by_condition(self):
        cases = load_jsonl("llm_demo.jsonl")
        run = self.submit(cases)
        expected = sum(len(c["prompts"]) for c in cases)
        self.assertEqual(run["total_tasks"], expected)
        self.assertEqual(run["status"], "queued")

    def test_idempotency_key_returns_same_run(self):
        run = self.submit(idempotency_key="abc")
        dataset_id = run["dataset_version_id"]
        again, created = repository.submit_run(self.conn, repository.RunRequest(
            dataset_version_id=dataset_id, provider="fixture", params={}, idempotency_key="abc"
        ), self.settings)
        self.assertFalse(created)
        self.assertEqual(run["id"], again["id"])


class LeaseTests(ServiceTestCase):
    def test_stale_worker_cannot_overwrite_new_attempt(self):
        run = self.submit(max_cases=1, conditions=("clean",))
        lease_a = self.claim("worker-a")
        # Simulate worker A stalling past its lease; the reaper requeues the task.
        self.conn.execute(
            "UPDATE inference_tasks SET lease_expires_at = now() - interval '1 second'"
        )
        self.conn.commit()
        self.assertEqual(queue.reap_expired_leases(self.conn), 1)
        lease_b = self.claim("worker-b")
        self.assertEqual(lease_b.task_id, lease_a.task_id)
        self.assertEqual(lease_b.attempt_no, 2)
        # A wakes up and tries to commit: rejected. B commits: accepted.
        self.assertFalse(queue.complete_task(self.conn, lease_a, self.output("stale")))
        self.assertTrue(queue.complete_task(self.conn, lease_b, self.output("fresh")))
        outputs = self.conn.execute("SELECT answer, attempt_id FROM model_outputs").fetchall()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["answer"], "fresh")
        self.assertEqual(outputs[0]["attempt_id"], lease_b.token)
        outcomes = sorted(r["outcome"] for r in self.conn.execute(
            "SELECT outcome FROM task_attempts").fetchall())
        self.assertEqual(outcomes, ["lease_expired", "succeeded"])
        self.assertEqual(repository.get_run(self.conn, run["id"])["status"], "succeeded")

    def test_heartbeat_extends_and_reports_lost_leases(self):
        self.submit(max_cases=1, conditions=("clean", "paraphrase"))
        kept, lost = self.claim(), self.claim()
        self.conn.execute(
            "UPDATE inference_tasks SET lease_token = NULL, status = 'queued' WHERE id = %s",
            (lost.task_id,),
        )
        self.conn.commit()
        self.assertEqual(queue.heartbeat(self.conn, [kept, lost], 30), {lost.token})

    def test_retryable_error_backs_off_then_fails_when_exhausted(self):
        run = self.submit(max_cases=1, conditions=("clean",), max_attempts=2)
        lease = self.claim()
        self.assertEqual(queue.fail_task(self.conn, lease, "server_error", "500", True, 0, 0), "retry_wait")
        lease = self.claim()
        self.assertEqual(lease.attempt_no, 2)
        self.assertEqual(queue.fail_task(self.conn, lease, "server_error", "500", True, 0, 0), "failed")
        self.assertIsNone(self.claim())
        self.assertEqual(repository.get_run(self.conn, run["id"])["status"], "failed")

    def test_fatal_error_is_not_retried(self):
        self.submit(max_cases=1, conditions=("clean",), max_attempts=5)
        lease = self.claim()
        self.assertEqual(queue.fail_task(self.conn, lease, "client_error", "404", False, 0, 0), "failed")

    def test_backoff_delays_retry(self):
        self.submit(max_cases=1, conditions=("clean",))
        lease = self.claim()
        queue.fail_task(self.conn, lease, "rate_limited", "429", True, 60, 60)
        # Full jitter may pick ~0, so assert on available_at being set, not on claimability.
        row = self.conn.execute("SELECT status, available_at > now() - interval '1 s' AS later"
                                " FROM inference_tasks").fetchone()
        self.assertEqual(row["status"], "retry_wait")
        self.assertTrue(row["later"])

    def test_partial_failure_status(self):
        run = self.submit(max_cases=1, conditions=("clean", "paraphrase"), max_attempts=1)
        first, second = self.claim(), self.claim()
        queue.complete_task(self.conn, first, self.output())
        queue.fail_task(self.conn, second, "timeout", "t", True, 0, 0)
        self.assertEqual(repository.get_run(self.conn, run["id"])["status"], "partially_failed")

    def test_cancel_stops_new_work_but_keeps_in_flight_output(self):
        run = self.submit(max_cases=2)
        in_flight = self.claim()
        result = queue.cancel_run(self.conn, run["id"])
        self.assertEqual(result["cancelled_tasks"], run["total_tasks"] - 1)
        self.assertIsNone(self.claim())
        self.assertEqual(repository.get_run(self.conn, run["id"])["status"], "running")
        self.assertTrue(queue.complete_task(self.conn, in_flight, self.output()))
        self.assertEqual(repository.get_run(self.conn, run["id"])["status"], "cancelled")


class ConcurrencyTests(ServiceTestCase):
    def test_provider_cap_is_global_across_connections(self):
        self.submit(provider="ollama", params={"model": "m", "base_url": "http://p"})
        repository.set_provider_limit(self.conn, "ollama:http://p", 2)
        other = connect(DB_URL)
        try:
            self.assertIsNotNone(self.claim("a"))
            self.assertIsNotNone(self.claim("b", conn=other))
            self.assertIsNone(self.claim("c"))
            self.assertIsNone(self.claim("d", conn=other))
        finally:
            other.close()

    def test_concurrent_claimers_never_share_a_task(self):
        self.submit(cases=load_jsonl("llm_benchmark_60.jsonl"))
        repository.set_provider_limit(self.conn, "fixture", 1000)
        claimed, lock = [], threading.Lock()

        def claimer(name):
            conn = connect(DB_URL)
            try:
                while True:
                    lease = queue.claim_task(conn, name, 30, 1000, "fair")
                    if lease is None:
                        return
                    with lock:
                        claimed.append(lease.task_id)
            finally:
                conn.close()

        threads = [threading.Thread(target=claimer, args=(f"c{i}",)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(claimed), 240)
        self.assertEqual(len(set(claimed)), 240)

    def test_fair_scheduler_interleaves_runs(self):
        big = self.submit(cases=load_jsonl("llm_benchmark_60.jsonl"))
        small_dataset = repository.create_dataset_version(self.conn, "small", load_jsonl("llm_demo.jsonl"))
        small, _ = repository.submit_run(self.conn, repository.RunRequest(
            dataset_version_id=small_dataset["id"], provider="fixture", params={}), self.settings)
        repository.set_provider_limit(self.conn, "fixture", 4)
        runs = [self.claim(f"w{i}").run_id for i in range(4)]
        self.assertEqual(runs.count(big["id"]), 2)
        self.assertEqual(runs.count(small["id"]), 2)

    def test_priority_beats_fairness(self):
        self.submit()
        dataset = repository.create_dataset_version(self.conn, "urgent", load_jsonl("llm_demo.jsonl")[:1])
        urgent, _ = repository.submit_run(self.conn, repository.RunRequest(
            dataset_version_id=dataset["id"], provider="fixture", params={}, priority=10), self.settings)
        self.assertEqual(self.claim().run_id, urgent["id"])

    def test_backpressure_rejects_when_queue_full(self):
        self.settings = settings(max_pending_tasks=10)
        with self.assertRaises(repository.BackpressureError):
            self.submit(cases=load_jsonl("llm_benchmark_60.jsonl"))
        self.submit(max_cases=2)  # 8 tasks fit
        with self.assertRaises(repository.BackpressureError):
            self.submit(max_cases=1)  # 8 + 4 > 10


class ScoringTests(ServiceTestCase):
    def run_fixture_to_completion(self):
        run = self.submit()
        while (lease := self.claim()) is not None:
            from shiftwatch.service.providers import build_provider
            out = build_provider("fixture", {}).generate(lease.prompt, lease.case_payload, lease.condition)
            queue.complete_task(self.conn, lease, out.as_record())
        return run

    def test_service_scoring_matches_batch_cli(self):
        from shiftwatch.llm_evaluation import evaluate_llm, fixture_model, summarize_llm
        run = self.run_fixture_to_completion()
        scoring_run, created = scoring.create_scoring_run(self.conn, run["id"])
        self.assertTrue(created)
        service = scoring.scoring_summary(self.conn, scoring_run["id"])["summary"]
        cases, model = fixture_model(ROOT / "datasets" / "llm_demo.jsonl")
        batch = summarize_llm(evaluate_llm(model, cases))
        self.assertEqual(service["conditions"], batch["conditions"])
        self.assertEqual(service["behavioral_consistency_rate"], batch["behavioral_consistency_rate"])

    def test_rescoring_is_versioned_and_never_calls_model(self):
        run = self.run_fixture_to_completion()
        attempts_before = self.conn.execute("SELECT count(*) AS n FROM task_attempts").fetchone()["n"]
        base, _ = scoring.create_scoring_run(self.conn, run["id"])
        again, created = scoring.create_scoring_run(self.conn, run["id"])
        self.assertFalse(created)
        self.assertEqual(base["id"], again["id"])
        case_id = load_jsonl("llm_demo.jsonl")[0]["id"]
        revised, _ = scoring.create_scoring_run(
            self.conn, run["id"], case_overrides={case_id: {"required_terms": ["zzz-never"]}}
        )
        self.assertNotEqual(base["id"], revised["id"])
        comparison = scoring.compare_scoring_runs(self.conn, [base["id"], revised["id"]])
        clean = comparison["by_condition"]["clean"]
        self.assertGreater(clean[str(base["id"])]["accuracy"], clean[str(revised["id"])]["accuracy"])
        attempts_after = self.conn.execute("SELECT count(*) AS n FROM task_attempts").fetchone()["n"]
        self.assertEqual(attempts_before, attempts_after)

    def test_scoring_requires_finished_run(self):
        run = self.submit()
        with self.assertRaises(repository.Conflict):
            scoring.create_scoring_run(self.conn, run["id"])


class ApiTests(ServiceTestCase):
    def setUp(self):
        super().setUp()
        self.client = TestClient(create_app(settings(max_pending_tasks=50)))

    def test_submit_poll_cancel_and_backpressure(self):
        dataset = self.client.post("/datasets", json={"name": "demo", "cases": load_jsonl("llm_demo.jsonl")})
        self.assertEqual(dataset.status_code, 201)
        self.assertEqual(self.client.post("/datasets", json={"name": "demo", "cases": load_jsonl("llm_demo.jsonl")}).status_code, 200)
        body = {"dataset_version_id": dataset.json()["id"], "provider": "fixture"}
        run = self.client.post("/runs", json=body, headers={"Idempotency-Key": "k1"})
        self.assertEqual(run.status_code, 202)
        self.assertEqual(self.client.post("/runs", json=body, headers={"Idempotency-Key": "k1"}).json()["id"], run.json()["id"])
        status = self.client.get(f"/runs/{run.json()['id']}").json()
        self.assertEqual(status["progress"], 0)
        page = self.client.get(f"/runs/{run.json()['id']}/tasks", params={"limit": 5}).json()
        self.assertEqual(len(page["items"]), 5)
        self.assertIsNotNone(page["next_after_id"])
        big = self.client.post("/datasets", json={"name": "big", "cases": load_jsonl("llm_benchmark_60.jsonl")})
        rejected = self.client.post("/runs", json={"dataset_version_id": big.json()["id"], "provider": "fixture"})
        self.assertEqual(rejected.status_code, 429)
        self.assertIn("Retry-After", rejected.headers)
        cancelled = self.client.post(f"/runs/{run.json()['id']}/cancel").json()
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.client.post(f"/runs/{run.json()['id']}/scorings", json={}).status_code, 201)

    def test_errors_map_to_http_codes(self):
        self.assertEqual(self.client.get("/runs/00000000-0000-0000-0000-000000000000").status_code, 404)
        self.assertEqual(self.client.post("/datasets", json={"name": "x", "cases": [{"id": 1}]}).status_code, 422)


class WorkerEndToEndTests(ServiceTestCase):
    def test_workers_complete_run_through_faulty_provider(self):
        server, state = serve_mock(latency_ms=5, jitter_ms=2, error_rate=0.15, drop_rate=0.05,
                                   invalid_json_rate=0.05, dataset=str(ROOT / "datasets" / "llm_benchmark_60.jsonl"))
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            run = self.submit(cases=load_jsonl("llm_benchmark_60.jsonl"), provider="ollama",
                              params={"model": "mock", "base_url": base_url, "timeout": 5},
                              max_attempts=6)
            repository.set_provider_limit(self.conn, f"ollama:{base_url}", 3)
            workers = [Worker(settings(lease_seconds=5), slots=4, poll_seconds=0.05) for _ in range(2)]
            for worker in workers:
                worker.start()
            deadline = time.time() + 60
            while time.time() < deadline:
                status = repository.get_run(self.conn, run["id"])
                self.conn.commit()
                if status["status"] not in ("queued", "running"):
                    break
                time.sleep(0.2)
            for worker in workers:
                worker.stop(timeout=10)
            self.assertEqual(status["status"], "succeeded", status["task_counts"])
            self.assertEqual(status["task_counts"], {"succeeded": 240})
            self.assertLessEqual(state.snapshot()["max_in_flight"], 3)
            parse_errors = self.conn.execute(
                "SELECT count(*) AS n FROM model_outputs WHERE parse_error IS NOT NULL").fetchone()["n"]
            self.assertGreater(parse_errors, 0)
            scoring_run, _ = scoring.create_scoring_run(self.conn, run["id"])
            self.assertEqual(scoring_run["scored_outputs"], 240)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
