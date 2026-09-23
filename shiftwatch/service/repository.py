"""Versioned inputs (datasets, model configs) and run submission/queries."""
from dataclasses import dataclass
import hashlib
import json

import psycopg
from psycopg.types.json import Jsonb

from ..llm_evaluation import case_from_dict

SUPPORTED_PROVIDERS = {"ollama", "fixture"}
ACTIVE_TASK_STATUSES = ("queued", "retry_wait", "running")


class BackpressureError(Exception):
    """The system is at capacity; the caller should retry later."""

    def __init__(self, pending: int, requested: int, limit: int):
        super().__init__(
            f"queue full: {pending} pending + {requested} requested > {limit}"
        )
        self.pending, self.requested, self.limit = pending, requested, limit


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


def canonical_sha256(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- datasets

def create_dataset_version(conn: psycopg.Connection, name: str, cases: list[dict]) -> dict:
    """Store an immutable dataset version. Identical content returns the existing row."""
    if not cases:
        raise ValueError("dataset must contain at least one case")
    seen = set()
    for index, item in enumerate(cases):
        try:
            case = case_from_dict(item)
        except (KeyError, TypeError) as error:
            raise ValueError(f"invalid case at index {index}: {error!r}") from error
        if not case.prompts:
            raise ValueError(f"case {case.id} has no prompts")
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id}")
        seen.add(case.id)
    digest = canonical_sha256(cases)
    with conn.transaction():
        row = conn.execute(
            "INSERT INTO dataset_versions (name, content_sha256, case_count)"
            " VALUES (%s, %s, %s) ON CONFLICT (name, content_sha256) DO NOTHING"
            " RETURNING id, name, content_sha256, case_count, created_at",
            (name, digest, len(cases)),
        ).fetchone()
        if row is None:
            existing = conn.execute(
                "SELECT id, name, content_sha256, case_count, created_at"
                " FROM dataset_versions WHERE name = %s AND content_sha256 = %s",
                (name, digest),
            ).fetchone()
            return {**existing, "created": False}
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO dataset_cases (dataset_version_id, case_id, position, payload)"
                " VALUES (%s, %s, %s, %s)",
                [(row["id"], item["id"], i, Jsonb(item)) for i, item in enumerate(cases)],
            )
    return {**row, "created": True}


def get_dataset_version(conn: psycopg.Connection, dataset_id) -> dict:
    row = conn.execute(
        "SELECT id, name, content_sha256, case_count, created_at"
        " FROM dataset_versions WHERE id = %s",
        (dataset_id,),
    ).fetchone()
    if row is None:
        raise NotFound(f"dataset version {dataset_id} not found")
    return row


# ---------------------------------------------------------------------- model configs

def provider_key_for(provider: str, params: dict) -> str:
    if provider == "ollama":
        return f"ollama:{params.get('base_url', 'http://localhost:11434').rstrip('/')}"
    return provider


def get_or_create_model_config(conn: psycopg.Connection, provider: str, params: dict) -> dict:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported provider {provider!r}")
    if provider == "ollama" and not params.get("model"):
        raise ValueError("ollama provider requires params.model")
    digest = canonical_sha256({"provider": provider, "params": params})
    key = provider_key_for(provider, params)
    with conn.transaction():
        conn.execute(
            "INSERT INTO model_configs (provider, provider_key, params, config_sha256)"
            " VALUES (%s, %s, %s, %s) ON CONFLICT (config_sha256) DO NOTHING",
            (provider, key, Jsonb(params), digest),
        )
        return conn.execute(
            "SELECT id, provider, provider_key, params, config_sha256, created_at"
            " FROM model_configs WHERE config_sha256 = %s",
            (digest,),
        ).fetchone()


def set_provider_limit(conn: psycopg.Connection, provider_key: str, max_concurrency: int) -> None:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    with conn.transaction():
        conn.execute(
            "INSERT INTO provider_limits (provider_key, max_concurrency) VALUES (%s, %s)"
            " ON CONFLICT (provider_key) DO UPDATE"
            " SET max_concurrency = EXCLUDED.max_concurrency, updated_at = now()",
            (provider_key, max_concurrency),
        )


# ---------------------------------------------------------------------------- runs

@dataclass(frozen=True)
class RunRequest:
    dataset_version_id: str
    provider: str
    params: dict
    label: str | None = None
    priority: int = 0
    max_attempts: int | None = None
    conditions: tuple[str, ...] | None = None  # None = every prompt condition
    max_cases: int | None = None
    idempotency_key: str | None = None


SUBMIT_LOCK_KEY = 72_110_416


def submit_run(conn: psycopg.Connection, request: RunRequest, settings) -> tuple[dict, bool]:
    """Create a run and one task per (case, condition). Returns (run, created).

    Each submission is a *new* inference experiment: prompts are never de-duplicated
    against earlier runs, because repeated sampling can be the point of the experiment.
    Use scoring runs to re-score stored outputs without calling the model.
    """
    model_config = get_or_create_model_config(conn, request.provider, request.params)
    max_attempts = request.max_attempts or settings.default_max_attempts
    with conn.transaction():
        if request.idempotency_key:
            existing = conn.execute(
                "SELECT * FROM inference_runs WHERE idempotency_key = %s",
                (request.idempotency_key,),
            ).fetchone()
            if existing:
                return existing, False
        get_dataset_version(conn, request.dataset_version_id)
        case_rows = conn.execute(
            "SELECT case_id, payload FROM dataset_cases WHERE dataset_version_id = %s"
            " ORDER BY position" + (" LIMIT %s" if request.max_cases else ""),
            (request.dataset_version_id, request.max_cases)
            if request.max_cases else (request.dataset_version_id,),
        ).fetchall()
        tasks = []
        for row in case_rows:
            for condition, prompt in row["payload"]["prompts"].items():
                if request.conditions is None or condition in request.conditions:
                    tasks.append((row["case_id"], condition, prompt))
        if not tasks:
            raise ValueError("run selects no tasks")
        if len(tasks) > settings.max_tasks_per_run:
            raise ValueError(
                f"run has {len(tasks)} tasks; limit is {settings.max_tasks_per_run}"
            )
        # Serialise admission so two concurrent submissions cannot both pass the check.
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (SUBMIT_LOCK_KEY,))
        pending = conn.execute(
            "SELECT count(*) AS n FROM inference_tasks WHERE status = ANY(%s)",
            (list(ACTIVE_TASK_STATUSES),),
        ).fetchone()["n"]
        if pending + len(tasks) > settings.max_pending_tasks:
            raise BackpressureError(pending, len(tasks), settings.max_pending_tasks)
        try:
            run = conn.execute(
                "INSERT INTO inference_runs (dataset_version_id, model_config_id, label,"
                " priority, max_attempts, total_tasks, idempotency_key)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (request.dataset_version_id, model_config["id"], request.label,
                 request.priority, max_attempts, len(tasks), request.idempotency_key),
            ).fetchone()
        except psycopg.errors.UniqueViolation as error:
            raise Conflict("idempotency key reused concurrently; retry") from error
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO inference_tasks (run_id, case_id, condition, prompt, max_attempts)"
                " VALUES (%s, %s, %s, %s, %s)",
                [(run["id"], c, cond, p, max_attempts) for c, cond, p in tasks],
            )
    return run, True


def get_run(conn: psycopg.Connection, run_id) -> dict:
    run = conn.execute(
        "SELECT r.*, mc.provider, mc.provider_key, mc.params AS model_params,"
        " dv.name AS dataset_name, dv.content_sha256 AS dataset_sha256"
        " FROM inference_runs r"
        " JOIN model_configs mc ON mc.id = r.model_config_id"
        " JOIN dataset_versions dv ON dv.id = r.dataset_version_id"
        " WHERE r.id = %s",
        (run_id,),
    ).fetchone()
    if run is None:
        raise NotFound(f"run {run_id} not found")
    counts = {
        row["status"]: row["n"]
        for row in conn.execute(
            "SELECT status, count(*) AS n FROM inference_tasks WHERE run_id = %s"
            " GROUP BY status",
            (run_id,),
        ).fetchall()
    }
    attempts = conn.execute(
        "SELECT count(*) AS attempts,"
        " count(*) FILTER (WHERE a.outcome = 'retryable_error') AS retryable_errors,"
        " count(*) FILTER (WHERE a.outcome = 'lease_expired') AS lease_expirations,"
        " count(*) FILTER (WHERE a.outcome = 'stale_discarded') AS stale_discards"
        " FROM task_attempts a JOIN inference_tasks t ON t.id = a.task_id"
        " WHERE t.run_id = %s",
        (run_id,),
    ).fetchone()
    terminal = sum(counts.get(s, 0) for s in ("succeeded", "failed", "cancelled"))
    run["task_counts"] = counts
    run["progress"] = terminal / run["total_tasks"]
    run["attempt_stats"] = attempts
    return run


def list_tasks(
    conn: psycopg.Connection,
    run_id,
    status: str | None = None,
    after_id: int = 0,
    limit: int = 50,
) -> list[dict]:
    """Keyset pagination on task id; stable under concurrent inserts, O(limit) per page."""
    limit = max(1, min(limit, 500))
    return conn.execute(
        "SELECT t.id, t.case_id, t.condition, t.status, t.attempt_count,"
        " t.last_error_kind, t.last_error, o.answer, o.confidence, o.abstain,"
        " o.parse_error, o.latency_ms"
        " FROM inference_tasks t LEFT JOIN model_outputs o ON o.task_id = t.id"
        " WHERE t.run_id = %s AND t.id > %s AND (%s::text IS NULL OR t.status = %s)"
        " ORDER BY t.id LIMIT %s",
        (run_id, after_id, status, status, limit),
    ).fetchall()


def list_runs(conn: psycopg.Connection, limit: int = 50) -> list[dict]:
    return conn.execute(
        "SELECT r.id, r.label, r.status, r.priority, r.total_tasks, r.created_at,"
        " r.finished_at, mc.provider_key FROM inference_runs r"
        " JOIN model_configs mc ON mc.id = r.model_config_id"
        " ORDER BY r.created_at DESC LIMIT %s",
        (max(1, min(limit, 200)),),
    ).fetchall()
