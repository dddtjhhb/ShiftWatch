"""PostgreSQL-backed task queue: claim, lease, heartbeat, complete, retry, reap, cancel.

Task state machine (inference_tasks.status):

    queued ──claim──▶ running ──complete──▶ succeeded
                        │
                        ├── retryable error, attempts left ──▶ retry_wait ──(available_at)──▶ claimable
                        ├── lease expired (worker died/hung) ─▶ retry_wait  (or failed if out of attempts)
                        ├── fatal error / attempts exhausted ─▶ failed
                        └── error or lease expiry after run cancelled ─▶ cancelled
    queued/retry_wait ──run cancelled──▶ cancelled

Guarantees and non-guarantees
  * At most one *effective* output per task: model_outputs.task_id is a primary key and
    an output is only inserted in the same transaction that flips the task from
    running→succeeded *while the caller's lease token is still current*.
  * A worker that lost its lease (e.g. paused past expiry, then resumed) gets
    ``stale`` back and its result is discarded and recorded as such.
  * External model calls are at-least-once, NOT exactly-once: if a worker dies after
    the provider answered but before commit, the retry calls the provider again.
    task_attempts records every claim so duplicate calls are measurable.
"""
from dataclasses import dataclass
import random
import uuid

import psycopg
from psycopg.types.json import Jsonb

from .repository import ACTIVE_TASK_STATUSES


@dataclass(frozen=True)
class Lease:
    task_id: int
    token: uuid.UUID  # == task_attempts.id
    attempt_no: int
    max_attempts: int
    run_id: uuid.UUID
    case_id: str
    condition: str
    prompt: str
    provider: str
    provider_key: str
    params: dict
    case_payload: dict


def _candidate_runs(conn, scheduler: str) -> list[dict]:
    """Runnable runs in scheduler order, at most 10 per provider.

    Limiting per provider (not globally) keeps many runs queued on one saturated
    provider from hiding runs that target an idle provider.
    """
    order = (
        "priority DESC, running ASC, last_claimed_at ASC NULLS FIRST, created_at ASC"
        if scheduler == "fair" else "priority DESC, created_at ASC"
    )
    return conn.execute(
        "SELECT id, provider_key FROM ("
        "  SELECT c.*, row_number() OVER (PARTITION BY provider_key ORDER BY"
        f"   {order}) AS rank_in_provider FROM ("
        "    SELECT r.id, r.priority, r.created_at, r.last_claimed_at, mc.provider_key,"
        "     (SELECT count(*) FROM inference_tasks t"
        "       WHERE t.run_id = r.id AND t.status = 'running') AS running"
        "    FROM inference_runs r JOIN model_configs mc ON mc.id = r.model_config_id"
        "    WHERE r.status IN ('queued', 'running') AND r.cancel_requested_at IS NULL"
        "      AND EXISTS (SELECT 1 FROM inference_tasks t WHERE t.run_id = r.id"
        "        AND t.status IN ('queued', 'retry_wait') AND t.available_at <= now())"
        "  ) c"
        ") ranked WHERE rank_in_provider <= 10"
        f" ORDER BY {order}"
    ).fetchall()


def claim_task(
    conn: psycopg.Connection,
    worker_id: str,
    lease_seconds: float,
    default_provider_concurrency: int,
    scheduler: str = "fair",
) -> Lease | None:
    """Claim one runnable task, honouring the global per-provider concurrency cap.

    The cap is enforced with a transaction-scoped advisory lock per provider key:
    claimers for the same provider serialise on it, count running tasks, and only
    claim below the cap. The lock is held until commit, so the next claimer's count
    already sees this claim. Each provider is tried in its own short transaction so
    a claimer never holds two provider locks (no lock-ordering deadlocks).

    Tasks whose worker crashed keep counting against the cap until their lease
    expires, which errs on the side of under-using capacity rather than exceeding it.
    """
    with conn.transaction():
        candidates = _candidate_runs(conn, scheduler)
    tried_providers: set[str] = set()
    for run in candidates:
        key = run["provider_key"]
        if key in tried_providers:
            continue
        tried_providers.add(key)
        # Runs for this provider, in scheduler order.
        run_ids = [r["id"] for r in candidates if r["provider_key"] == key]
        lease = _claim_for_provider(
            conn, key, run_ids, worker_id, lease_seconds, default_provider_concurrency
        )
        if lease is not None:
            return lease
    return None


def _claim_for_provider(
    conn, key, run_ids, worker_id, lease_seconds, default_provider_concurrency
) -> Lease | None:
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,))
        limit_row = conn.execute(
            "SELECT max_concurrency FROM provider_limits WHERE provider_key = %s", (key,)
        ).fetchone()
        limit = limit_row["max_concurrency"] if limit_row else default_provider_concurrency
        running = conn.execute(
            "SELECT count(*) AS n FROM inference_tasks t"
            " JOIN inference_runs r ON r.id = t.run_id"
            " JOIN model_configs mc ON mc.id = r.model_config_id"
            " WHERE t.status = 'running' AND mc.provider_key = %s",
            (key,),
        ).fetchone()["n"]
        if running >= limit:
            return None
        for run_id in run_ids:
            task = conn.execute(
                "SELECT t.id FROM inference_tasks t"
                " JOIN inference_runs r ON r.id = t.run_id"
                " WHERE t.run_id = %s AND t.status IN ('queued', 'retry_wait')"
                "   AND t.available_at <= now() AND r.cancel_requested_at IS NULL"
                # Lowest id first: requeued/retried tasks resume ahead of untouched
                # ones, so recovery after a crash is not pushed to the end of the run.
                " ORDER BY t.id FOR UPDATE OF t SKIP LOCKED LIMIT 1",
                (run_id,),
            ).fetchone()
            if task is not None:
                break
        else:
            return None
        token = uuid.uuid4()
        row = conn.execute(
            "UPDATE inference_tasks SET status = 'running',"
            " attempt_count = attempt_count + 1, lease_token = %s, lease_owner = %s,"
            " lease_expires_at = now() + make_interval(secs => %s), updated_at = now()"
            " WHERE id = %s RETURNING id, run_id, case_id, condition, prompt,"
            " attempt_count, max_attempts",
            (token, worker_id, lease_seconds, task["id"]),
        ).fetchone()
        conn.execute(
            "INSERT INTO task_attempts (id, task_id, attempt_no, worker_id, provider_key)"
            " VALUES (%s, %s, %s, %s, %s)",
            (token, row["id"], row["attempt_count"], worker_id, key),
        )
        conn.execute(
            "UPDATE inference_runs SET last_claimed_at = clock_timestamp(),"
            " status = CASE WHEN status = 'queued' THEN 'running' ELSE status END,"
            " started_at = coalesce(started_at, now())"
            " WHERE id = %s",
            (row["run_id"],),
        )
        context = conn.execute(
            "SELECT mc.provider, mc.params, dc.payload FROM inference_runs r"
            " JOIN model_configs mc ON mc.id = r.model_config_id"
            " JOIN dataset_cases dc ON dc.dataset_version_id = r.dataset_version_id"
            "   AND dc.case_id = %s"
            " WHERE r.id = %s",
            (row["case_id"], row["run_id"]),
        ).fetchone()
    return Lease(
        task_id=row["id"], token=token, attempt_no=row["attempt_count"],
        max_attempts=row["max_attempts"], run_id=row["run_id"],
        case_id=row["case_id"], condition=row["condition"], prompt=row["prompt"],
        provider=context["provider"], provider_key=key,
        params=context["params"], case_payload=context["payload"],
    )


def heartbeat(conn: psycopg.Connection, leases: list[Lease], lease_seconds: float) -> set[uuid.UUID]:
    """Extend every lease still held. Returns the tokens that are no longer valid."""
    if not leases:
        return set()
    tokens = [lease.token for lease in leases]
    with conn.transaction():
        renewed = {
            row["lease_token"]
            for row in conn.execute(
                "UPDATE inference_tasks"
                " SET lease_expires_at = now() + make_interval(secs => %s), updated_at = now()"
                " WHERE lease_token = ANY(%s) AND status = 'running'"
                " RETURNING lease_token",
                (lease_seconds, tokens),
            ).fetchall()
        }
    return set(tokens) - renewed


def _finish_attempt(conn, token, outcome, error_kind=None, error=None) -> None:
    conn.execute(
        "UPDATE task_attempts SET finished_at = now(), outcome = %s, error_kind = %s,"
        " error = %s WHERE id = %s AND outcome IS NULL",
        (outcome, error_kind, error, token),
    )


def complete_task(conn: psycopg.Connection, lease: Lease, output: dict) -> bool:
    """Store the attempt's output iff it still owns the lease. False means stale."""
    with conn.transaction():
        row = conn.execute(
            "UPDATE inference_tasks SET status = 'succeeded', lease_token = NULL,"
            " lease_owner = NULL, lease_expires_at = NULL, last_error_kind = NULL,"
            " last_error = NULL, updated_at = now()"
            " WHERE id = %s AND lease_token = %s AND status = 'running' RETURNING run_id",
            (lease.task_id, lease.token),
        ).fetchone()
        if row is None:
            _finish_attempt(conn, lease.token, "stale_discarded")
            return False
        conn.execute(
            "INSERT INTO model_outputs (task_id, attempt_id, raw_text, answer, confidence,"
            " abstain, parse_error, latency_ms, usage)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (lease.task_id, lease.token, output["raw_text"], output.get("answer"),
             output.get("confidence"), output.get("abstain"), output.get("parse_error"),
             output["latency_ms"], Jsonb(output.get("usage", {}))),
        )
        _finish_attempt(conn, lease.token, "succeeded")
        finalize_run_if_done(conn, row["run_id"])
    return True


def backoff_seconds(attempt_no: int, base: float, cap: float, rng=random) -> float:
    """Exponential backoff with full jitter."""
    return rng.uniform(0, min(cap, base * (2 ** (attempt_no - 1))))


def fail_task(
    conn: psycopg.Connection,
    lease: Lease,
    error_kind: str,
    error: str,
    retryable: bool,
    retry_base_seconds: float,
    retry_max_seconds: float,
) -> str | None:
    """Record a failed attempt. Returns the new task status, or None if the lease was stale."""
    will_retry = retryable and lease.attempt_no < lease.max_attempts
    delay = backoff_seconds(lease.attempt_no, retry_base_seconds, retry_max_seconds)
    wanted = "retry_wait" if will_retry else "failed"
    with conn.transaction():
        # A retry of a cancelled run would never be claimed again (and the run would
        # never finish), so it becomes 'cancelled' instead.
        row = conn.execute(
            "UPDATE inference_tasks t SET status = CASE"
            "   WHEN %s = 'retry_wait' AND r.cancel_requested_at IS NOT NULL THEN 'cancelled'"
            "   ELSE %s END,"
            " lease_token = NULL, lease_owner = NULL,"
            " lease_expires_at = NULL, available_at = now() + make_interval(secs => %s),"
            " last_error_kind = %s, last_error = %s, updated_at = now()"
            " FROM inference_runs r"
            " WHERE t.id = %s AND t.lease_token = %s AND t.status = 'running'"
            "   AND r.id = t.run_id RETURNING t.run_id, t.status",
            (wanted, wanted, delay, error_kind, error[:2000], lease.task_id, lease.token),
        ).fetchone()
        if row is None:
            _finish_attempt(conn, lease.token, "stale_discarded", error_kind, error[:2000])
            return None
        _finish_attempt(
            conn, lease.token, "retryable_error" if retryable else "fatal_error",
            error_kind, error[:2000],
        )
        if row["status"] != "retry_wait":
            finalize_run_if_done(conn, row["run_id"])
    return row["status"]


def reap_expired_leases(conn: psycopg.Connection) -> int:
    """Return tasks whose lease expired to the queue (or fail them if out of attempts).

    An expired attempt counts as used: the provider may already have been called.
    """
    with conn.transaction():
        rows = conn.execute(
            "WITH expired AS ("
            "  SELECT id, lease_token FROM inference_tasks"
            "  WHERE status = 'running' AND lease_expires_at < now()"
            "  FOR UPDATE SKIP LOCKED)"
            " UPDATE inference_tasks t SET"
            "  status = CASE WHEN t.attempt_count >= t.max_attempts THEN 'failed'"
            "                WHEN r.cancel_requested_at IS NOT NULL THEN 'cancelled'"
            "                ELSE 'retry_wait' END,"
            "  lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL,"
            "  available_at = now(), last_error_kind = 'lease_expired',"
            "  last_error = 'worker lease expired', updated_at = now()"
            " FROM expired, inference_runs r"
            " WHERE t.id = expired.id AND r.id = t.run_id"
            " RETURNING t.run_id, t.status, expired.lease_token AS old_token"
        ).fetchall()
        for row in rows:
            _finish_attempt(conn, row["old_token"], "lease_expired", "lease_expired")
        for run_id in {row["run_id"] for row in rows if row["status"] != "retry_wait"}:
            finalize_run_if_done(conn, run_id)
    return len(rows)


def cancel_run(conn: psycopg.Connection, run_id) -> dict:
    """Stop scheduling new work for a run.

    Queued/retrying tasks are cancelled immediately. Requests already sent to a
    provider cannot be recalled (and are already paid for), so running tasks are
    allowed to finish and their outputs are kept; the run becomes 'cancelled' once
    nothing is running.

    Lock order: every other path locks task rows before the run row (claim,
    complete, fail, reap). Cancelling therefore commits the cancel flag in its own
    short transaction first, then cancels tasks and finalises (task → run order).
    Once the flag is committed, claims skip the run and any retry or lease expiry
    of its in-flight tasks turns into 'cancelled', so no task can be stranded.
    """
    with conn.transaction():
        run = conn.execute(
            "UPDATE inference_runs"
            " SET cancel_requested_at = coalesce(cancel_requested_at, now())"
            " WHERE id = %s AND status IN ('queued', 'running') RETURNING id",
            (run_id,),
        ).fetchone()
    if run is None:
        existing = conn.execute(
            "SELECT status FROM inference_runs WHERE id = %s", (run_id,)
        ).fetchone()
        conn.commit()
        if existing is None:
            return {}
        return {"status": existing["status"], "cancelled_tasks": 0}
    with conn.transaction():
        cancelled = conn.execute(
            "UPDATE inference_tasks SET status = 'cancelled', updated_at = now()"
            " WHERE run_id = %s AND status IN ('queued', 'retry_wait')",
            (run_id,),
        ).rowcount
        status = finalize_run_if_done(conn, run_id)
    return {"status": status, "cancelled_tasks": cancelled}


def finalize_run_if_done(conn: psycopg.Connection, run_id) -> str:
    """Derive the run's terminal status once no task is active. Caller holds a transaction."""
    run = conn.execute(
        "SELECT status, cancel_requested_at FROM inference_runs WHERE id = %s FOR UPDATE",
        (run_id,),
    ).fetchone()
    counts = {
        row["status"]: row["n"]
        for row in conn.execute(
            "SELECT status, count(*) AS n FROM inference_tasks WHERE run_id = %s"
            " GROUP BY status",
            (run_id,),
        ).fetchall()
    }
    if any(counts.get(status, 0) for status in ACTIVE_TASK_STATUSES):
        return run["status"]
    if run["status"] not in ("queued", "running"):
        return run["status"]
    succeeded = counts.get("succeeded", 0)
    failed = counts.get("failed", 0)
    if run["cancel_requested_at"] is not None:
        status = "cancelled"
    elif failed == 0:
        status = "succeeded"
    elif succeeded == 0:
        status = "failed"
    else:
        status = "partially_failed"
    conn.execute(
        "UPDATE inference_runs SET status = %s, finished_at = now() WHERE id = %s",
        (status, run_id),
    )
    return status
