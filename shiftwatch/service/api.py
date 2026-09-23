"""HTTP API. Submission returns immediately with a run id; workers do the work."""
from contextlib import contextmanager
import uuid

from fastapi import FastAPI, Header, HTTPException, Query, Response
import psycopg
from pydantic import BaseModel, Field

from . import queue, repository, scoring
from .config import Settings
from .db import connect


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    cases: list[dict] = Field(min_length=1)


class RunIn(BaseModel):
    dataset_version_id: uuid.UUID
    provider: str = "ollama"
    params: dict = Field(default_factory=dict)
    label: str | None = None
    priority: int = Field(default=0, ge=-100, le=100)
    max_attempts: int | None = Field(default=None, ge=1, le=20)
    conditions: list[str] | None = None
    max_cases: int | None = Field(default=None, ge=1)


class ScoringIn(BaseModel):
    rubric: str = "keyword_v1"
    confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    case_overrides: dict[str, dict] = Field(default_factory=dict)


class ProviderLimitIn(BaseModel):
    max_concurrency: int = Field(ge=1, le=1000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="ShiftWatch evaluation service", version="0.2.0")

    @contextmanager
    def db():
        # One short-lived connection per request keeps the service dependency-light;
        # swap in psycopg_pool if connection setup ever shows up in latency profiles.
        with connect(settings.database_url) as conn:
            try:
                yield conn
            except repository.NotFound as error:
                raise HTTPException(404, str(error)) from error
            except repository.Conflict as error:
                raise HTTPException(409, str(error)) from error
            except psycopg.errors.UniqueViolation as error:
                raise HTTPException(409, "concurrent duplicate request; retry") from error
            except ValueError as error:
                raise HTTPException(422, str(error)) from error

    @app.get("/healthz")
    def healthz():
        with db() as conn:
            conn.execute("SELECT 1")
        return {"ok": True}

    @app.post("/datasets", status_code=201)
    def create_dataset(body: DatasetIn, response: Response):
        with db() as conn:
            row = repository.create_dataset_version(conn, body.name, body.cases)
        if not row["created"]:
            response.status_code = 200
        return row

    @app.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: uuid.UUID):
        with db() as conn:
            return repository.get_dataset_version(conn, dataset_id)

    @app.post("/runs", status_code=202)
    def submit_run(
        body: RunIn,
        response: Response,
        idempotency_key: str | None = Header(default=None, max_length=200),
    ):
        request = repository.RunRequest(
            dataset_version_id=str(body.dataset_version_id), provider=body.provider,
            params=body.params, label=body.label, priority=body.priority,
            max_attempts=body.max_attempts,
            conditions=tuple(body.conditions) if body.conditions else None,
            max_cases=body.max_cases, idempotency_key=idempotency_key,
        )
        try:
            with db() as conn:
                run, created = repository.submit_run(conn, request, settings)
        except repository.BackpressureError as error:
            raise HTTPException(
                429, str(error), headers={"Retry-After": "30"}
            ) from error
        if not created:
            response.status_code = 200
        response.headers["Location"] = f"/runs/{run['id']}"
        return run

    @app.get("/runs")
    def list_runs(limit: int = Query(50, ge=1, le=200)):
        with db() as conn:
            return repository.list_runs(conn, limit)

    @app.get("/runs/{run_id}")
    def get_run(run_id: uuid.UUID):
        with db() as conn:
            return repository.get_run(conn, run_id)

    @app.get("/runs/{run_id}/tasks")
    def list_tasks(
        run_id: uuid.UUID,
        status: str | None = None,
        after_id: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=500),
    ):
        with db() as conn:
            repository.get_run(conn, run_id)
            items = repository.list_tasks(conn, run_id, status, after_id, limit)
        next_after = items[-1]["id"] if len(items) == limit else None
        return {"items": items, "next_after_id": next_after}

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: uuid.UUID):
        with db() as conn:
            result = queue.cancel_run(conn, run_id)
        if not result:
            raise HTTPException(404, f"run {run_id} not found")
        return result

    @app.post("/runs/{run_id}/scorings", status_code=201)
    def create_scoring(run_id: uuid.UUID, body: ScoringIn, response: Response):
        with db() as conn:
            row, created = scoring.create_scoring_run(
                conn, run_id, body.rubric, body.confidence_threshold, body.case_overrides
            )
        if not created:
            response.status_code = 200
        return row

    @app.get("/scorings/{scoring_id}")
    def get_scoring(scoring_id: uuid.UUID):
        with db() as conn:
            return scoring.scoring_summary(conn, scoring_id)

    @app.get("/compare")
    def compare(scoring_ids: list[uuid.UUID] = Query(...)):
        with db() as conn:
            return scoring.compare_scoring_runs(conn, scoring_ids)

    @app.put("/providers/{provider_key:path}/limit")
    def set_limit(provider_key: str, body: ProviderLimitIn):
        with db() as conn:
            repository.set_provider_limit(conn, provider_key, body.max_concurrency)
        return {"provider_key": provider_key, "max_concurrency": body.max_concurrency}

    @app.get("/stats")
    def stats():
        with db() as conn:
            by_status = {
                r["status"]: r["n"] for r in conn.execute(
                    "SELECT status, count(*) AS n FROM inference_tasks GROUP BY status"
                ).fetchall()
            }
            running_by_provider = {
                r["provider_key"]: r["n"] for r in conn.execute(
                    "SELECT mc.provider_key, count(*) AS n FROM inference_tasks t"
                    " JOIN inference_runs r ON r.id = t.run_id"
                    " JOIN model_configs mc ON mc.id = r.model_config_id"
                    " WHERE t.status = 'running' GROUP BY mc.provider_key"
                ).fetchall()
            }
        pending = sum(by_status.get(s, 0) for s in repository.ACTIVE_TASK_STATUSES)
        return {
            "tasks_by_status": by_status,
            "pending_tasks": pending,
            "queue_capacity": settings.max_pending_tasks,
            "running_by_provider": running_by_provider,
        }

    return app
