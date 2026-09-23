-- ShiftWatch evaluation service: initial schema.
--
-- Design rules
--   * Inputs are immutable and content-addressed (dataset_versions, model_configs).
--   * Inference (expensive, non-deterministic) and scoring (cheap, deterministic)
--     live in separate tables, so a rubric change never requires re-inference and
--     never overwrites an earlier score.
--   * Every execution attempt gets its own row and token. Only the attempt that
--     still holds the task lease may write the task's single effective output.
--   * All lease arithmetic uses the database clock (now()), never worker clocks.

CREATE TABLE dataset_versions (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name            text NOT NULL,
    content_sha256  text NOT NULL,
    case_count      integer NOT NULL CHECK (case_count > 0),
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (name, content_sha256)
);

CREATE TABLE dataset_cases (
    dataset_version_id uuid NOT NULL REFERENCES dataset_versions(id),
    case_id            text NOT NULL,
    position           integer NOT NULL,
    payload            jsonb NOT NULL,
    PRIMARY KEY (dataset_version_id, case_id)
);

CREATE TABLE model_configs (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    provider        text NOT NULL,          -- ollama | fixture
    provider_key    text NOT NULL,          -- concurrency-limit bucket, e.g. ollama:http://host:11434
    params          jsonb NOT NULL,         -- model name, base_url, temperature, seed, ...
    config_sha256   text NOT NULL UNIQUE,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Global (cross-worker) concurrency cap per provider endpoint.
CREATE TABLE provider_limits (
    provider_key    text PRIMARY KEY,
    max_concurrency integer NOT NULL CHECK (max_concurrency > 0),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE inference_runs (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_version_id  uuid NOT NULL REFERENCES dataset_versions(id),
    model_config_id     uuid NOT NULL REFERENCES model_configs(id),
    label               text,
    status              text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'succeeded', 'partially_failed',
                          'failed', 'cancelled')),
    priority            integer NOT NULL DEFAULT 0,
    max_attempts        integer NOT NULL CHECK (max_attempts BETWEEN 1 AND 20),
    total_tasks         integer NOT NULL,
    idempotency_key     text UNIQUE,
    cancel_requested_at timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now(),
    started_at          timestamptz,
    finished_at         timestamptz
);
CREATE INDEX inference_runs_active_idx ON inference_runs (status)
    WHERE status IN ('queued', 'running');

CREATE TABLE inference_tasks (
    id                bigserial PRIMARY KEY,
    run_id            uuid NOT NULL REFERENCES inference_runs(id),
    case_id           text NOT NULL,
    condition         text NOT NULL,
    prompt            text NOT NULL,
    status            text NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'retry_wait', 'succeeded',
                          'failed', 'cancelled')),
    attempt_count     integer NOT NULL DEFAULT 0,
    max_attempts      integer NOT NULL,
    available_at      timestamptz NOT NULL DEFAULT now(),
    lease_token       uuid,
    lease_owner       text,
    lease_expires_at  timestamptz,
    last_error_kind   text,
    last_error        text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (run_id, case_id, condition),
    -- a running task always has a lease; a non-running task never does
    CHECK ((status = 'running') = (lease_token IS NOT NULL))
);
CREATE INDEX inference_tasks_claim_idx ON inference_tasks (run_id, id)
    WHERE status IN ('queued', 'retry_wait');
CREATE INDEX inference_tasks_running_idx ON inference_tasks (lease_expires_at)
    WHERE status = 'running';
CREATE INDEX inference_tasks_run_status_idx ON inference_tasks (run_id, status);

-- One row per claim. id doubles as the lease token held by that attempt.
CREATE TABLE task_attempts (
    id           uuid PRIMARY KEY,
    task_id      bigint NOT NULL REFERENCES inference_tasks(id),
    attempt_no   integer NOT NULL,
    worker_id    text NOT NULL,
    provider_key text NOT NULL,
    claimed_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    outcome      text CHECK (outcome IN ('succeeded', 'retryable_error', 'fatal_error',
                                         'lease_expired', 'stale_discarded')),
    error_kind   text,
    error        text,
    UNIQUE (task_id, attempt_no)
);

-- Exactly one effective output per task (primary key), written only by the
-- attempt that held the lease at commit time.
CREATE TABLE model_outputs (
    task_id      bigint PRIMARY KEY REFERENCES inference_tasks(id),
    attempt_id   uuid NOT NULL REFERENCES task_attempts(id),
    raw_text     text NOT NULL,
    answer       text,
    confidence   double precision,
    abstain      boolean,
    parse_error  text,              -- model replied, but not in the JSON contract
    latency_ms   integer NOT NULL,
    usage        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE scoring_runs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    inference_run_id  uuid NOT NULL REFERENCES inference_runs(id),
    rubric            text NOT NULL,
    rubric_sha256     text NOT NULL,
    params            jsonb NOT NULL,
    scored_outputs    integer NOT NULL,
    missing_outputs   integer NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (inference_run_id, rubric_sha256)
);

CREATE TABLE scores (
    scoring_run_id uuid NOT NULL REFERENCES scoring_runs(id),
    task_id        bigint NOT NULL REFERENCES inference_tasks(id),
    case_id        text NOT NULL,
    category       text NOT NULL,
    condition      text NOT NULL,
    correct        boolean NOT NULL,
    details        jsonb NOT NULL,
    PRIMARY KEY (scoring_run_id, task_id)
);
CREATE INDEX scores_condition_idx ON scores (scoring_run_id, condition);
